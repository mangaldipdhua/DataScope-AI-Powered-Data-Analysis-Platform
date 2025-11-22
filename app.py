"""
DataScope - AI-Powered Data Analysis Platform
A Flask web application for CSV data analysis with AI insights
"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session
from werkzeug.utils import secure_filename
import google.generativeai as genai
from datetime import datetime
import io
import base64
from pathlib import Path
import logging
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from functools import lru_cache
import hashlib

try:
    from mongo_utils import db
except ImportError:
    db = None

generate_chat_response = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_change_in_production')

# Configuration
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static/plots'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create directories
for folder in [UPLOAD_FOLDER, PLOTS_FOLDER]:
    Path(folder).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Configure Gemini AI
try:
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini AI configured successfully")
except Exception as e:
    logger.error(f"Gemini AI configuration failed: {e}")
    model = None

# Configure matplotlib for better plots
plt.style.use('dark_background')
sns.set_palette("husl")

# Startup Information
logger.info("\n" + "="*70)
logger.info("DataScope Application Startup")
logger.info("="*70)
logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
logger.info(f"Debug Mode: {os.getenv('DEBUG', 'False')}")
logger.info(f"Flask Env: {os.getenv('FLASK_ENV', 'production')}")

if model:
    logger.info("✓ Gemini AI: Ready")
else:
    logger.info("✗ Gemini AI: Failed to initialize")

logger.info(f"✓ MongoDB Status: {'Connected' if db.connected else 'Not Connected (app will work without it)'}")



logger.info("="*70 + "\n")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_timestamp():
    """Generate timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Convert both keys and values
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'dtype'):  # Any numpy type we might have missed
        return obj.item() if hasattr(obj, 'item') else str(obj)
    elif str(type(obj)).startswith("<class 'numpy."):  # Catch any remaining numpy types
        return str(obj)
    else:
        return obj

def safe_read_file(filepath):
    """Safely read CSV/Excel files with encoding detection"""
    try:
        if filepath.suffix.lower() == '.csv':
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    return pd.read_csv(filepath, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode file with any supported encoding")
        else:
            return pd.read_excel(filepath)
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        raise

def clean_data(df):
    """Basic data cleaning"""
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Convert numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                df[col] = numeric_series
    
    return df

def generate_data_analysis(df, timestamp):
    """Generate comprehensive data analysis and visualizations"""
    analysis = {
        'basic_info': {},
        'statistics': {},
        'missing_data': {},
        'data_quality': {},
        'feature_analysis': {},
        'outlier_analysis': {},
        'correlation_analysis': {},
        'distribution_analysis': {},
        'insights': {},
        'plots': {}
    }
    
    # Basic information
    analysis['basic_info'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # Statistical summary
    analysis['statistics'] = df.describe(include='all').to_dict()
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    analysis['missing_data'] = {
        'missing_counts': missing_data.to_dict(),
        'missing_percentages': (missing_data / len(df) * 100).to_dict()
    }
    
    # Data quality assessment
    analysis['data_quality'] = {
        'total_missing': missing_data.sum(),
        'missing_percentage': (missing_data.sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'duplicates': df.duplicated().sum(),
        'unique_values_per_column': df.nunique().to_dict(),
        'unique_values': len(df.nunique()),
        'data_types_count': df.dtypes.value_counts().to_dict()
    }
    
    # 🔍 1. Basic Descriptive Statistics
    try:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    except Exception as e:
        logger.error(f"Error in column type detection: {e}")
        numeric_columns = []
        categorical_columns = list(df.columns)
    
    analysis['descriptive_stats'] = {}
    try:
        if numeric_columns:
            desc_stats = df[numeric_columns].describe()
            analysis['descriptive_stats'] = {
                'basic': desc_stats.to_dict(),
                'skewness': df[numeric_columns].skew().to_dict(),
                'kurtosis': df[numeric_columns].kurtosis().to_dict(),
                'variance': df[numeric_columns].var().to_dict()
            }
    except Exception as e:
        logger.error(f"Error in descriptive statistics: {e}")
        analysis['descriptive_stats'] = {}
    
    # 📊 2. Data Types & Distribution Analysis
    try:
        feature_types = {
            'continuous': [],
            'discrete': [],
            'binary': [],
            'high_cardinality': []
        }
        
        analysis['distribution_analysis'] = {
            'numeric_features': len(numeric_columns),
            'categorical_features': len(categorical_columns),
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'feature_types': feature_types
        }
        
        # Analyze feature types
        for col in numeric_columns:
            unique_vals = df[col].nunique()
            if unique_vals == 2:
                feature_types['binary'].append(col)
            elif unique_vals < 10:
                feature_types['discrete'].append(col)
            else:
                feature_types['continuous'].append(col)
        
        for col in categorical_columns:
            unique_vals = df[col].nunique()
            if unique_vals > 50:
                feature_types['high_cardinality'].append(col)
    except Exception as e:
        logger.error(f"Error in distribution analysis: {e}")
        analysis['distribution_analysis'] = {
            'numeric_features': 0,
            'categorical_features': 0,
            'numeric_columns': [],
            'categorical_columns': [],
            'feature_types': {'continuous': [], 'discrete': [], 'binary': [], 'high_cardinality': []}
        }
    
    # 🧱 4. Correlation Analysis
    analysis['correlation_analysis'] = {}
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        analysis['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': [],
            'multicollinearity_risk': []
        }
        
        # Find high correlations
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    analysis['correlation_analysis']['high_correlations'].append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
                if abs(corr_val) > 0.9:
                    analysis['correlation_analysis']['multicollinearity_risk'].append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
    
    # 📉 3. Missing Data Analysis (Enhanced)
    analysis['missing_analysis'] = {
        'missing_patterns': {},
        'columns_with_missing': [],
        'missing_severity': 'low'
    }
    
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        if missing_pct > 0:
            analysis['missing_analysis']['columns_with_missing'].append({
                'column': col,
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': missing_pct
            })
    
    if analysis['data_quality']['missing_percentage'] > 20:
        analysis['missing_analysis']['missing_severity'] = 'high'
    elif analysis['data_quality']['missing_percentage'] > 5:
        analysis['missing_analysis']['missing_severity'] = 'medium'
    
    # 📉 5. Outlier Detection
    analysis['outlier_analysis'] = {}
    try:
        if numeric_columns:
            outliers = {}
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(df)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            
            analysis['outlier_analysis'] = outliers
    except Exception as e:
        logger.error(f"Error in outlier analysis: {e}")
        analysis['outlier_analysis'] = {}
    
    # ⚖️ 6. Feature Importance / Variance Analysis
    analysis['variance_analysis'] = {}
    try:
        if numeric_columns:
            variances = df[numeric_columns].var()
            analysis['variance_analysis'] = {
                'low_variance_features': [],
                'zero_variance_features': [],
                'feature_variances': variances.to_dict()
            }
            
            for col, var_val in variances.items():
                if var_val == 0:
                    analysis['variance_analysis']['zero_variance_features'].append(col)
                elif var_val < 0.01:
                    analysis['variance_analysis']['low_variance_features'].append(col)
    except Exception as e:
        logger.error(f"Error in variance analysis: {e}")
        analysis['variance_analysis'] = {
            'low_variance_features': [],
            'zero_variance_features': [],
            'feature_variances': {}
        }
    
    # 🧮 7. Cardinality Analysis
    analysis['cardinality_analysis'] = {}
    for col in categorical_columns:
        unique_count = df[col].nunique()
        analysis['cardinality_analysis'][col] = {
            'unique_count': unique_count,
            'cardinality_level': 'high' if unique_count > 50 else 'medium' if unique_count > 10 else 'low',
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    # 🧪 9. Feature Interaction Insights
    analysis['feature_interactions'] = {
        'potential_interactions': [],
        'categorical_numeric_pairs': []
    }
    
    # Find potential interactions between categorical and numeric features
    for cat_col in categorical_columns[:3]:  # Limit to first 3 to avoid too many combinations
        for num_col in numeric_columns[:3]:
            if df[cat_col].nunique() < 10:  # Only for low cardinality categorical
                analysis['feature_interactions']['categorical_numeric_pairs'].append({
                    'categorical': cat_col,
                    'numeric': num_col,
                    'categories': df[cat_col].nunique()
                })
    
    # 🧠 10. Domain-Specific Checks
    analysis['data_validation'] = {
        'negative_values': {},
        'constant_columns': [],
        'potential_issues': []
    }
    
    # Check for negative values in columns that shouldn't have them
    for col in numeric_columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            analysis['data_validation']['negative_values'][col] = negative_count
    
    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            analysis['data_validation']['constant_columns'].append(col)
    
    # Check for potential ID columns
    for col in df.columns:
        if df[col].nunique() == len(df) and 'id' in col.lower():
            analysis['data_validation']['potential_issues'].append(f"{col} appears to be an ID column")
    
    # Memory usage analysis
    try:
        analysis['memory_analysis'] = {
            'total_memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'memory_per_column': (df.memory_usage(deep=True) / 1024 / 1024).to_dict()
        }
    except Exception as e:
        logger.error(f"Error in memory analysis: {e}")
        analysis['memory_analysis'] = {
            'total_memory_mb': 0.0,
            'memory_per_column': {}
        }
    
    # Generate insights
    insights = []
    
    # Data size insights
    if df.shape[0] > 10000:
        insights.append(f"Large dataset with {df.shape[0]:,} rows - suitable for robust statistical analysis")
    elif df.shape[0] < 100:
        insights.append(f"Small dataset with {df.shape[0]} rows - results may have limited statistical power")
    
    # Missing data insights
    if analysis['data_quality']['missing_percentage'] > 20:
        insights.append("High percentage of missing data detected - consider data cleaning strategies")
    elif analysis['data_quality']['missing_percentage'] == 0:
        insights.append("Complete dataset with no missing values - excellent data quality")
    
    # Duplicate data insights
    if analysis['data_quality']['duplicate_rows'] > 0:
        insights.append(f"Found {analysis['data_quality']['duplicate_rows']} duplicate rows - consider deduplication")
    
    analysis['insights'] = insights
    
    # Advanced Feature Analysis
    try:
        # Import scipy.stats here to ensure it's available
        from scipy import stats as scipy_stats
        
        # Separate column types for detailed analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Feature type analysis
        analysis['feature_analysis'] = {
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'datetime_features': len(datetime_cols),
            'feature_types': {
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'datetime': datetime_cols
            },
            'skewness': {},
            'kurtosis': {},
            'normality_tests': {}
        }
        
        # Statistical analysis for numeric features
        for col in numeric_cols:
            if df[col].notna().sum() > 1:
                data = df[col].dropna()
                try:
                    analysis['feature_analysis']['skewness'][col] = float(scipy_stats.skew(data))
                    analysis['feature_analysis']['kurtosis'][col] = float(scipy_stats.kurtosis(data))
                    
                    # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
                    if len(data) <= 5000:
                        try:
                            stat, p_value = scipy_stats.shapiro(data)
                            analysis['feature_analysis']['normality_tests'][col] = {
                                'test': 'Shapiro-Wilk',
                                'statistic': float(stat),
                                'p_value': float(p_value),
                                'is_normal': p_value > 0.05
                            }
                        except:
                            analysis['feature_analysis']['normality_tests'][col] = {'test': 'Failed', 'is_normal': False}
                except Exception as e:
                    logger.warning(f"Statistical analysis failed for column {col}: {e}")
        
        # Outlier Analysis using IQR and Isolation Forest
        analysis['outlier_analysis'] = {
            'iqr_outliers': {},
            'z_score_outliers': {},
            'isolation_forest_outliers': {}
        }
        
        for col in numeric_cols:
            if df[col].notna().sum() > 1:
                data = df[col].dropna()
                
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
                analysis['outlier_analysis']['iqr_outliers'][col] = len(outliers_iqr)
                
                # Z-score method
                try:
                    z_scores = np.abs(scipy_stats.zscore(data))
                    outliers_z = data[z_scores > 3]
                    analysis['outlier_analysis']['z_score_outliers'][col] = len(outliers_z)
                except:
                    analysis['outlier_analysis']['z_score_outliers'][col] = 0
        
        # Isolation Forest for multivariate outliers
        if len(numeric_cols) > 1:
            try:
                numeric_data = df[numeric_cols].dropna()
                if len(numeric_data) > 10:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers_iso = iso_forest.fit_predict(numeric_data)
                    analysis['outlier_analysis']['isolation_forest_outliers']['total'] = int(np.sum(outliers_iso == -1))
            except:
                analysis['outlier_analysis']['isolation_forest_outliers']['total'] = 0
        
        # Correlation Analysis
        analysis['correlation_analysis'] = {
            'pearson_correlations': {},
            'spearman_correlations': {},
            'mutual_information': {},
            'cramers_v': {}
        }
        
        # Pearson and Spearman correlations for numeric features
        if len(numeric_cols) > 1:
            numeric_data = df[numeric_cols]
            analysis['correlation_analysis']['pearson_correlations'] = numeric_data.corr().to_dict()
            analysis['correlation_analysis']['spearman_correlations'] = numeric_data.corr(method='spearman').to_dict()
        
        # Cramér's V for categorical features
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2 / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
        if len(categorical_cols) > 1:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    try:
                        if df[col1].notna().sum() > 0 and df[col2].notna().sum() > 0:
                            cramers = cramers_v(df[col1].dropna(), df[col2].dropna())
                            analysis['correlation_analysis']['cramers_v'][f"{col1}_vs_{col2}"] = float(cramers)
                    except:
                        pass
        
        # Distribution Analysis
        analysis['distribution_analysis'] = {
            'numeric_distributions': {},
            'categorical_distributions': {}
        }
        
        for col in numeric_cols:
            if df[col].notna().sum() > 1:
                data = df[col].dropna()
                analysis['distribution_analysis']['numeric_distributions'][col] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'unique_values': int(data.nunique()),
                    'zero_count': int((data == 0).sum()),
                    'negative_count': int((data < 0).sum())
                }
        
        for col in categorical_cols:
            if df[col].notna().sum() > 0:
                value_counts = df[col].value_counts()
                analysis['distribution_analysis']['categorical_distributions'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A',
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'least_frequent': str(value_counts.index[-1]) if len(value_counts) > 0 else 'N/A',
                    'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                    'distribution': value_counts.head(10).to_dict()
                }
    
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")
        # Initialize empty analysis sections if they failed
        if 'feature_analysis' not in analysis:
            analysis['feature_analysis'] = {
                'numeric_features': 0,
                'categorical_features': 0,
                'datetime_features': 0,
                'feature_types': {'numeric': [], 'categorical': [], 'datetime': []},
                'skewness': {},
                'kurtosis': {},
                'normality_tests': {}
            }
        if 'outlier_analysis' not in analysis:
            analysis['outlier_analysis'] = {
                'iqr_outliers': {},
                'z_score_outliers': {},
                'isolation_forest_outliers': {}
            }
        if 'correlation_analysis' not in analysis:
            analysis['correlation_analysis'] = {
                'pearson_correlations': {},
                'spearman_correlations': {},
                'mutual_information': {},
                'cramers_v': {}
            }
        if 'distribution_analysis' not in analysis:
            analysis['distribution_analysis'] = {
                'numeric_distributions': {},
                'categorical_distributions': {}
            }
    
    # Generate visualizations with proper error handling
    try:
        # Separate numeric and categorical columns properly
        numeric_columns = []
        categorical_columns = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if column has meaningful variation
                if df[col].nunique() > 1 and not df[col].isnull().all():
                    try:
                        std_val = df[col].std()
                        if pd.notna(std_val) and std_val > 1e-10:  # Very small threshold for variance
                            numeric_columns.append(col)
                        else:
                            logger.warning(f"Skipping numeric column {col}: std={std_val}")
                    except:
                        logger.warning(f"Error calculating std for {col}")
                else:
                    logger.warning(f"Skipping column {col}: unique_values={df[col].nunique()}")
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                if df[col].nunique() > 1 and df[col].nunique() < len(df) * 0.8:  # Not too many unique values
                    categorical_columns.append(col)
        
        logger.info(f"Valid numeric columns: {numeric_columns}")
        logger.info(f"Valid categorical columns: {categorical_columns}")
        
        # Set up beautiful color schemes and plotting style
        plt.style.use('default')
        
        # Define beautiful color palettes
        primary_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316']
        gradient_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
        pastel_colors = ['#a8e6cf', '#dcedc1', '#ffd3a5', '#ffa8a8', '#b4a7d6', '#96ceb4', '#feca57', '#ff9ff3']
        
        # Set seaborn style for better plots
        sns.set_style("whitegrid")
        sns.set_palette(primary_colors)
        
    except Exception as e:
        logger.error(f"Error in column analysis: {e}")
        numeric_columns = []
        categorical_columns = []
    
    # 1. Data Overview Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        
        # Dataset info text
        info_text = f"""Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns
Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
Numeric Columns: {len(numeric_columns)}
Categorical Columns: {len(categorical_columns)}"""
        
        axes[0, 0].text(0.1, 0.5, info_text, transform=axes[0, 0].transAxes, 
                        fontsize=12, color='#2d3748', verticalalignment='center')
        axes[0, 0].set_title('Dataset Information', color='#2d3748', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Missing data heatmap
        try:
            if df.isnull().sum().sum() > 0:
                sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=axes[0, 1])
                axes[0, 1].set_title('Missing Data Pattern', color='#2d3748', fontsize=14, fontweight='bold')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', 
                               transform=axes[0, 1].transAxes, fontsize=16, color='#2d3748')
                axes[0, 1].set_title('Missing Data Pattern', color='#2d3748', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
        except Exception as e:
            logger.warning(f"Missing data heatmap failed: {e}")
            axes[0, 1].text(0.5, 0.5, 'Missing Data\nVisualization\nUnavailable', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=12, color='#2d3748')
            axes[0, 1].axis('off')
        
        # Data types pie chart
        try:
            dtype_counts = df.dtypes.value_counts()
            if len(dtype_counts) > 0:
                wedges, texts, autotexts = axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, 
                                                         autopct='%1.1f%%', colors=['#2d3748', '#a9a29c', '#333333'])
                # Set text colors
                for text in texts:
                    text.set_color('#2d3748')
                for autotext in autotexts:
                    autotext.set_color('white')
                axes[1, 0].set_title('Data Types Distribution', color='#2d3748', fontsize=14, fontweight='bold')
        except Exception as e:
            logger.warning(f"Data types pie chart failed: {e}")
            axes[1, 0].axis('off')
        
        # Data completeness
        try:
            completeness = (1 - df.isnull().sum() / len(df)) * 100
            if len(completeness) > 0:
                completeness.plot(kind='bar', ax=axes[1, 1], color='#a9a29c')
                axes[1, 1].set_title('Data Completeness by Column', color='#2d3748', fontsize=14, fontweight='bold')
                axes[1, 1].set_ylabel('Completeness %', color='#2d3748')
                axes[1, 1].tick_params(colors='#2d3748')
                # Rotate x-axis labels if too many columns
                if len(completeness) > 10:
                    axes[1, 1].tick_params(axis='x', rotation=45)
        except Exception as e:
            logger.warning(f"Completeness chart failed: {e}")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        overview_path = f"{PLOTS_FOLDER}/overview_{timestamp}.png"
        plt.savefig(overview_path, facecolor='white', dpi=300, bbox_inches='tight')
        plt.close()
        analysis['plots']['overview'] = f"overview_{timestamp}.png"
        
    except Exception as e:
        logger.error(f"Overview plot failed: {e}")
        if 'fig' in locals():
            plt.close(fig)
    
    # 2. Correlation Matrix (for numeric data)
    if len(numeric_columns) > 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.patch.set_facecolor('white')
            
            # Create correlation matrix from numeric columns
            correlation_matrix = df[numeric_columns].corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Create heatmap
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                       center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
            
            ax.set_title('Correlation Matrix', color='#2d3748', fontsize=16, fontweight='bold', pad=20)
            
            # Set tick colors
            ax.tick_params(colors='#2d3748')
            
            plt.tight_layout()
            corr_path = f"{PLOTS_FOLDER}/correlation_{timestamp}.png"
            plt.savefig(corr_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['correlation'] = f"correlation_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 3. Distribution Analysis
    if len(numeric_columns) > 0:
        try:
            n_plots = min(len(numeric_columns), 4)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.patch.set_facecolor('white')
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_columns[:4]):
                try:
                    # Create histogram with beautiful colors
                    df[col].hist(bins=min(30, df[col].nunique()), alpha=0.8, ax=axes[i], 
                                color=primary_colors[i % len(primary_colors)], density=True)
                    
                    # Try to add KDE if data is suitable
                    try:
                        if df[col].nunique() > 5:  # Need enough unique values for KDE
                            df[col].plot.kde(ax=axes[i], color=gradient_colors[i % len(gradient_colors)], linewidth=3)
                    except Exception as kde_error:
                        logger.warning(f"KDE failed for column {col}: {kde_error}")
                    
                    axes[i].set_title(f'Distribution of {col}', color='#2d3748', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(col, color='#2d3748')
                    axes[i].set_ylabel('Density', color='#2d3748')
                    axes[i].tick_params(colors='#2d3748')
                    
                except Exception as e:
                    logger.warning(f"Distribution plot failed for {col}: {e}")
                    # Create simple text plot as fallback
                    axes[i].text(0.5, 0.5, f'Distribution plot\nunavailable for\n{col}', 
                               ha='center', va='center', transform=axes[i].transAxes, 
                               color='#2d3748', fontsize=10)
                    axes[i].set_title(f'Distribution of {col}', color='#2d3748', fontsize=12, fontweight='bold')
            
            # Hide unused subplots
            for i in range(len(numeric_columns), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            dist_path = f"{PLOTS_FOLDER}/distributions_{timestamp}.png"
            plt.savefig(dist_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['distributions'] = f"distributions_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Distribution plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 4. Box Plots for Outlier Detection
    if len(numeric_columns) > 0:
        try:
            n_numeric = len(numeric_columns)
            n_cols = min(n_numeric, 3)
            n_rows = max(1, (n_numeric + n_cols - 1) // n_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig.patch.set_facecolor('white')
            
            # Handle axes properly for different subplot configurations
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            else:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            for i, col in enumerate(numeric_columns):
                try:
                    ax = axes[i]
                    
                    # Use seaborn boxplot with beautiful colors
                    sns.boxplot(data=df, y=col, ax=ax, color=pastel_colors[i % len(pastel_colors)])
                    
                    ax.set_title(f'Box Plot - {col}', color='#2d3748', fontsize=12, fontweight='bold')
                    ax.set_ylabel(col, color='#2d3748')
                    ax.tick_params(colors='#2d3748')
                    
                except Exception as e:
                    logger.warning(f"Box plot failed for {col}: {e}")
                    # Create text fallback
                    ax = axes[i]
                    ax.text(0.5, 0.5, f'Box plot\nunavailable for\n{col}', 
                           ha='center', va='center', transform=ax.transAxes, 
                           color='#2d3748', fontsize=10)
                    ax.set_title(f'Box Plot - {col}', color='#2d3748', fontsize=12, fontweight='bold')
                    ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_numeric, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            box_path = f"{PLOTS_FOLDER}/boxplots_{timestamp}.png"
            plt.savefig(box_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['boxplots'] = f"boxplots_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Box plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 5. Categorical Data Analysis
    if len(categorical_columns) > 0:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.patch.set_facecolor('white')
            axes = axes.flatten()
            
            for i, col in enumerate(categorical_columns[:4]):
                try:
                    if df[col].nunique() < 50:  # Only plot if not too many categories
                        value_counts = df[col].value_counts().head(10)
                        if len(value_counts) > 0:
                            value_counts.plot(kind='bar', ax=axes[i], color=primary_colors[i % len(primary_colors)])
                            axes[i].set_title(f'Distribution of {col}', color='#2d3748', fontsize=12, fontweight='bold')
                            axes[i].set_xlabel(col, color='#2d3748')
                            axes[i].set_ylabel('Count', color='#2d3748')
                            axes[i].tick_params(colors='#2d3748')
                            axes[i].tick_params(axis='x', rotation=45)
                        else:
                            axes[i].axis('off')
                    else:
                        # Too many categories, show summary
                        axes[i].text(0.5, 0.5, f'{col}\n{df[col].nunique()} unique values\n(Too many to plot)', 
                                   ha='center', va='center', transform=axes[i].transAxes, 
                                   color='#2d3748', fontsize=10)
                        axes[i].set_title(f'Distribution of {col}', color='#2d3748', fontsize=12, fontweight='bold')
                        axes[i].axis('off')
                except Exception as e:
                    logger.warning(f"Categorical plot failed for {col}: {e}")
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(categorical_columns), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            cat_path = f"{PLOTS_FOLDER}/categorical_{timestamp}.png"
            plt.savefig(cat_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['categorical'] = f"categorical_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Categorical plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 4. Statistical Distribution Analysis
    if len(numeric_columns) > 0:
        try:
            n_cols = min(3, len(numeric_columns))
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig.patch.set_facecolor('white')
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    try:
                        # Create histogram with KDE
                        data = df[col].dropna()
                        if len(data) > 1:
                            axes[i].hist(data, bins=30, alpha=0.7, color='#a9a29c', density=True)
                            
                            # Add KDE if possible
                            try:
                                from scipy import stats
                                kde = stats.gaussian_kde(data)
                                x_range = np.linspace(data.min(), data.max(), 100)
                                axes[i].plot(x_range, kde(x_range), color='#2d3748', linewidth=2)
                            except:
                                pass
                            
                            axes[i].set_title(f'{col} Distribution', color='#2d3748', fontweight='bold')
                            axes[i].set_xlabel(col, color='#2d3748')
                            axes[i].set_ylabel('Density', color='#2d3748')
                            axes[i].tick_params(colors='#2d3748')
                            
                            # Add statistics text
                            mean_val = data.mean()
                            std_val = data.std()
                            axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                            axes[i].legend()
                    except Exception as e:
                        logger.warning(f"Distribution plot failed for {col}: {e}")
                        axes[i].text(0.5, 0.5, f'Distribution\nfor {col}\nUnavailable', 
                                   ha='center', va='center', transform=axes[i].transAxes, 
                                   fontsize=12, color='#2d3748')
                        axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(numeric_columns), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            dist_path = f"{PLOTS_FOLDER}/distributions_{timestamp}.png"
            plt.savefig(dist_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['statistical_distributions'] = f"distributions_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Distribution plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 5. Data Quality Summary Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor('white')
        
        # Missing data by column
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_counts[missing_counts > 0].plot(kind='bar', ax=axes[0, 0], color='#ef4444')
            axes[0, 0].set_title('Missing Values by Column', color='#2d3748', fontweight='bold')
            axes[0, 0].set_ylabel('Missing Count', color='#2d3748')
            axes[0, 0].tick_params(colors='#2d3748', axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=16, color='#2d3748')
            axes[0, 0].set_title('Missing Values by Column', color='#2d3748', fontweight='bold')
            axes[0, 0].axis('off')
        
        # Unique values per column
        unique_counts = df.nunique()
        unique_counts.plot(kind='bar', ax=axes[0, 1], color='#10b981')
        axes[0, 1].set_title('Unique Values per Column', color='#2d3748', fontweight='bold')
        axes[0, 1].set_ylabel('Unique Count', color='#2d3748')
        axes[0, 1].tick_params(colors='#2d3748', axis='x', rotation=45)
        
        # Data types distribution
        dtype_counts = df.dtypes.value_counts()
        wedges, texts, autotexts = axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, 
                                                 autopct='%1.1f%%', colors=['#2d3748', '#a9a29c', '#333333'])
        for text in texts:
            text.set_color('#2d3748')
        for autotext in autotexts:
            autotext.set_color('white')
        axes[1, 0].set_title('Data Types Distribution', color='#2d3748', fontweight='bold')
        
        # Memory usage by column
        memory_usage = df.memory_usage(deep=True)
        memory_usage.plot(kind='bar', ax=axes[1, 1], color='#f59e0b')
        axes[1, 1].set_title('Memory Usage by Column', color='#2d3748', fontweight='bold')
        axes[1, 1].set_ylabel('Memory (bytes)', color='#2d3748')
        axes[1, 1].tick_params(colors='#2d3748', axis='x', rotation=45)
        
        plt.tight_layout()
        quality_path = f"{PLOTS_FOLDER}/data_quality_{timestamp}.png"
        plt.savefig(quality_path, facecolor='white', dpi=300, bbox_inches='tight')
        plt.close()
        analysis['plots']['data_quality'] = f"data_quality_{timestamp}.png"
        
    except Exception as e:
        logger.error(f"Data quality plots failed: {e}")
        if 'fig' in locals():
            plt.close(fig)
    
    # 6. Advanced Statistical Analysis (if applicable)
    if len(numeric_columns) >= 2:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.patch.set_facecolor('white')
            
            # Box plots for outlier detection
            df[numeric_columns].boxplot(ax=axes[0], patch_artist=True)
            axes[0].set_title('Outlier Detection (Box Plots)', color='#2d3748', fontweight='bold')
            axes[0].tick_params(colors='#2d3748', axis='x', rotation=45)
            axes[0].set_ylabel('Values', color='#2d3748')
            
            # Scatter plot matrix (for first few numeric columns)
            if len(numeric_columns) >= 2:
                col1, col2 = numeric_columns[0], numeric_columns[1]
                axes[1].scatter(df[col1], df[col2], alpha=0.6, color='#a9a29c')
                axes[1].set_xlabel(col1, color='#2d3748')
                axes[1].set_ylabel(col2, color='#2d3748')
                axes[1].set_title(f'{col1} vs {col2}', color='#2d3748', fontweight='bold')
                axes[1].tick_params(colors='#2d3748')
                
                # Add correlation coefficient
                corr = df[col1].corr(df[col2])
                axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=axes[1].transAxes, color='#2d3748', 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            advanced_path = f"{PLOTS_FOLDER}/advanced_stats_{timestamp}.png"
            plt.savefig(advanced_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['advanced_stats'] = f"advanced_stats_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Advanced statistical plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 7. Comprehensive Missing Value Analysis
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        # Missing value heatmap
        if df.isnull().sum().sum() > 0:
            sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=axes[0, 0])
            axes[0, 0].set_title('Missing Value Pattern', color='#2d3748', fontweight='bold')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=16, color='#2d3748')
            axes[0, 0].set_title('Missing Value Pattern', color='#2d3748', fontweight='bold')
            axes[0, 0].axis('off')
        
        # Missing value counts
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_counts[missing_counts > 0].plot(kind='bar', ax=axes[0, 1], color='#a9a29c')
            axes[0, 1].set_title('Missing Values by Column', color='#2d3748', fontweight='bold')
            axes[0, 1].set_ylabel('Missing Count', color='#2d3748')
            axes[0, 1].tick_params(colors='#2d3748', axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=16, color='#2d3748')
            axes[0, 1].axis('off')
        
        # Missing value percentages
        missing_pct = (df.isnull().sum() / len(df) * 100)
        if missing_pct.sum() > 0:
            missing_pct[missing_pct > 0].plot(kind='bar', ax=axes[1, 0], color='#2d3748')
            axes[1, 0].set_title('Missing Values Percentage', color='#2d3748', fontweight='bold')
            axes[1, 0].set_ylabel('Missing %', color='#2d3748')
            axes[1, 0].tick_params(colors='#2d3748', axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=16, color='#2d3748')
            axes[1, 0].axis('off')
        
        # Data completeness matrix
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        completeness.plot(kind='barh', ax=axes[1, 1], color='#a9a29c')
        axes[1, 1].set_title('Data Completeness by Column', color='#2d3748', fontweight='bold')
        axes[1, 1].set_xlabel('Completeness %', color='#2d3748')
        axes[1, 1].tick_params(colors='#2d3748')
        
        plt.tight_layout()
        missing_path = f"{PLOTS_FOLDER}/missing_analysis_{timestamp}.png"
        plt.savefig(missing_path, facecolor='white', dpi=300, bbox_inches='tight')
        plt.close()
        analysis['plots']['missing_analysis'] = f"missing_analysis_{timestamp}.png"
        
    except Exception as e:
        logger.error(f"Missing value analysis plots failed: {e}")
        if 'fig' in locals():
            plt.close(fig)
    
    # 8. Outlier Detection Visualization
    if len(numeric_columns) > 0:
        try:
            n_cols = min(3, len(numeric_columns))
            n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            fig.patch.set_facecolor('white')
            
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_columns):
                if i < len(axes):
                    try:
                        data = df[col].dropna()
                        if len(data) > 1:
                            # Box plot with outlier highlighting
                            bp = axes[i].boxplot(data, patch_artist=True, 
                                               boxprops=dict(facecolor='#a9a29c', alpha=0.7),
                                               medianprops=dict(color='#2d3748', linewidth=2),
                                               whiskerprops=dict(color='#2d3748'),
                                               capprops=dict(color='#2d3748'),
                                               flierprops=dict(marker='o', markerfacecolor='red', 
                                                             markersize=5, alpha=0.7))
                            
                            axes[i].set_title(f'{col} - Outlier Detection', color='#2d3748', fontweight='bold')
                            axes[i].set_ylabel('Values', color='#2d3748')
                            axes[i].tick_params(colors='#2d3748')
                            
                            # Add outlier statistics
                            Q1 = data.quantile(0.25)
                            Q3 = data.quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                            axes[i].text(0.02, 0.98, f'Outliers: {len(outliers)}', 
                                       transform=axes[i].transAxes, color='#2d3748', 
                                       verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except Exception as e:
                        logger.warning(f"Outlier plot failed for {col}: {e}")
                        axes[i].text(0.5, 0.5, f'Outlier Analysis\nfor {col}\nUnavailable', 
                                   ha='center', va='center', transform=axes[i].transAxes, 
                                   fontsize=12, color='#2d3748')
                        axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(numeric_columns), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            outlier_path = f"{PLOTS_FOLDER}/outlier_detection_{timestamp}.png"
            plt.savefig(outlier_path, facecolor='white', dpi=300, bbox_inches='tight')
            plt.close()
            analysis['plots']['outlier_detection'] = f"outlier_detection_{timestamp}.png"
            
        except Exception as e:
            logger.error(f"Outlier detection plots failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 9. Feature Relationship Analysis
    if len(numeric_columns) >= 2:
        try:
            # Pair plot for numeric features (limit to first 5 for performance)
            plot_cols = numeric_columns[:5]
            if len(plot_cols) >= 2:
                fig, axes = plt.subplots(len(plot_cols), len(plot_cols), figsize=(15, 15))
                fig.patch.set_facecolor('white')
                
                for i, col1 in enumerate(plot_cols):
                    for j, col2 in enumerate(plot_cols):
                        if i == j:
                            # Diagonal: histogram
                            data = df[col1].dropna()
                            axes[i, j].hist(data, bins=20, alpha=0.7, color='#a9a29c')
                            axes[i, j].set_title(f'{col1}', color='#2d3748', fontsize=10)
                        else:
                            # Off-diagonal: scatter plot
                            data1 = df[col1].dropna()
                            data2 = df[col2].dropna()
                            common_idx = data1.index.intersection(data2.index)
                            if len(common_idx) > 1:
                                axes[i, j].scatter(df.loc[common_idx, col1], df.loc[common_idx, col2], 
                                                 alpha=0.6, color='#a9a29c', s=20)
                                
                                # Add correlation coefficient
                                try:
                                    corr = df[col1].corr(df[col2])
                                    axes[i, j].text(0.05, 0.95, f'r={corr:.3f}', 
                                                   transform=axes[i, j].transAxes, color='#2d3748', 
                                                   fontsize=8,
                                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                                except:
                                    pass
                        
                        axes[i, j].tick_params(colors='#2d3748', labelsize=8)
                        if i == len(plot_cols) - 1:
                            axes[i, j].set_xlabel(col2, color='#2d3748', fontsize=9)
                        if j == 0:
                            axes[i, j].set_ylabel(col1, color='#2d3748', fontsize=9)
                
                plt.tight_layout()
                pair_path = f"{PLOTS_FOLDER}/pair_plot_{timestamp}.png"
                plt.savefig(pair_path, facecolor='white', dpi=300, bbox_inches='tight')
                plt.close()
                analysis['plots']['pair_plot'] = f"pair_plot_{timestamp}.png"
        
        except Exception as e:
            logger.error(f"Pair plot failed: {e}")
            if 'fig' in locals():
                plt.close(fig)
    
    # 10. Statistical Summary Visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor('white')
        
        # Skewness analysis
        if len(numeric_columns) > 0:
            skewness_data = []
            skewness_labels = []
            for col in numeric_columns:
                if col in analysis.get('feature_analysis', {}).get('skewness', {}):
                    skewness_data.append(analysis['feature_analysis']['skewness'][col])
                    skewness_labels.append(col)
            
            if skewness_data:
                bars = axes[0, 0].bar(range(len(skewness_data)), skewness_data, color='#a9a29c')
                axes[0, 0].set_title('Skewness by Feature', color='#2d3748', fontweight='bold')
                axes[0, 0].set_ylabel('Skewness', color='#2d3748')
                axes[0, 0].set_xticks(range(len(skewness_labels)))
                axes[0, 0].set_xticklabels(skewness_labels, rotation=45, color='#2d3748')
                axes[0, 0].tick_params(colors='#2d3748')
                axes[0, 0].axhline(y=0, color='#2d3748', linestyle='--', alpha=0.5)
                
                # Color bars based on skewness level
                for i, (bar, skew) in enumerate(zip(bars, skewness_data)):
                    if abs(skew) > 1:
                        bar.set_color('#ef4444')  # High skewness - red
                    elif abs(skew) > 0.5:
                        bar.set_color('#f59e0b')  # Moderate skewness - orange
                    else:
                        bar.set_color('#10b981')  # Low skewness - green
        
        # Kurtosis analysis
        if len(numeric_columns) > 0:
            kurtosis_data = []
            kurtosis_labels = []
            for col in numeric_columns:
                if col in analysis.get('feature_analysis', {}).get('kurtosis', {}):
                    kurtosis_data.append(analysis['feature_analysis']['kurtosis'][col])
                    kurtosis_labels.append(col)
            
            if kurtosis_data:
                bars = axes[0, 1].bar(range(len(kurtosis_data)), kurtosis_data, color='#a9a29c')
                axes[0, 1].set_title('Kurtosis by Feature', color='#2d3748', fontweight='bold')
                axes[0, 1].set_ylabel('Kurtosis', color='#2d3748')
                axes[0, 1].set_xticks(range(len(kurtosis_labels)))
                axes[0, 1].set_xticklabels(kurtosis_labels, rotation=45, color='#2d3748')
                axes[0, 1].tick_params(colors='#2d3748')
                axes[0, 1].axhline(y=0, color='#2d3748', linestyle='--', alpha=0.5)
        
        # Outlier summary
        if 'outlier_analysis' in analysis:
            outlier_methods = ['iqr_outliers', 'z_score_outliers']
            method_names = ['IQR Method', 'Z-Score Method']
            
            for method, name in zip(outlier_methods, method_names):
                if method in analysis['outlier_analysis']:
                    outlier_counts = list(analysis['outlier_analysis'][method].values())
                    if outlier_counts:
                        axes[1, 0].bar(range(len(outlier_counts)), outlier_counts, 
                                     alpha=0.7, label=name)
            
            if len(numeric_columns) > 0:
                axes[1, 0].set_title('Outlier Count by Method', color='#2d3748', fontweight='bold')
                axes[1, 0].set_ylabel('Outlier Count', color='#2d3748')
                axes[1, 0].set_xticks(range(len(numeric_columns)))
                axes[1, 0].set_xticklabels(numeric_columns, rotation=45, color='#2d3748')
                axes[1, 0].tick_params(colors='#2d3748')
                axes[1, 0].legend()
        
        # Feature type distribution
        feature_types = ['Numeric', 'Categorical', 'DateTime']
        feature_counts = [
            analysis.get('feature_analysis', {}).get('numeric_features', 0),
            analysis.get('feature_analysis', {}).get('categorical_features', 0),
            analysis.get('feature_analysis', {}).get('datetime_features', 0)
        ]
        
        if sum(feature_counts) > 0:
            wedges, texts, autotexts = axes[1, 1].pie(feature_counts, labels=feature_types, 
                                                     autopct='%1.1f%%', colors=['#2d3748', '#a9a29c', '#333333'])
            for text in texts:
                text.set_color('#2d3748')
            for autotext in autotexts:
                autotext.set_color('white')
            axes[1, 1].set_title('Feature Types Distribution', color='#2d3748', fontweight='bold')
        
        plt.tight_layout()
        stats_path = f"{PLOTS_FOLDER}/statistical_summary_{timestamp}.png"
        plt.savefig(stats_path, facecolor='white', dpi=300, bbox_inches='tight')
        plt.close()
        analysis['plots']['statistical_summary'] = f"statistical_summary_{timestamp}.png"
        
    except Exception as e:
        logger.error(f"Statistical summary plots failed: {e}")
        if 'fig' in locals():
            plt.close(fig)
    
    return analysis

def get_ai_insights(df, analysis):
    """Generate AI insights using Gemini"""
    if not model:
        return "AI insights unavailable - Gemini API not configured"
    
    try:
        # Prepare data summary for AI
        summary = f"""
        Dataset Analysis Summary:
        - Shape: {analysis['basic_info']['shape']}
        - Columns: {', '.join(analysis['basic_info']['columns'][:10])}{'...' if len(analysis['basic_info']['columns']) > 10 else ''}
        - Data Types: {analysis['basic_info']['dtypes']}
        - Missing Data: {analysis['missing_data']['missing_percentages']}
        - Key Statistics: {str(analysis['statistics'])[:1000]}
        """
        
        prompt = f"""
        As a data scientist, analyze this dataset and provide insights:
        
        {summary}
        
        Please provide:
        1. Key findings and patterns
        2. Data quality assessment
        3. Recommendations for further analysis
        4. Potential use cases
        5. Data cleaning suggestions
        
        Keep the response concise and actionable.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"AI insight generation failed: {e}")
        return f"AI insights unavailable: {str(e)}"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze uploaded file"""
    try:
        if 'file' not in request.files:
            error_msg = 'No file selected'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
                return jsonify({'success': False, 'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('upload_page'))
        
        file = request.files['file']
        if file.filename == '':
            error_msg = 'No file selected'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
                return jsonify({'success': False, 'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('upload_page'))
        
        if not allowed_file(file.filename):
            error_msg = 'Invalid file type. Please upload CSV or Excel files only.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
                return jsonify({'success': False, 'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('upload_page'))
        
        # Save uploaded file
        timestamp = generate_timestamp()
        filename = secure_filename(file.filename)
        safe_filename = f"{timestamp}_{filename}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / safe_filename
        file.save(filepath)
        
        try:
            if db and db.connected:
                file_size = os.path.getsize(filepath)
                upload_data = {
                    'filename': file.filename,
                    'file_size': file_size,
                    'file_type': file.content_type,
                    'upload_path': str(filepath)
                }
                db.save_upload_info(filename, upload_data)
        except Exception as e:
            logger.warning(f"Failed to save upload metadata to MongoDB: {e}")
        
        # Read and analyze data
        df = safe_read_file(filepath)
        df = clean_data(df)
        
        if df.empty:
            error_msg = 'The uploaded file appears to be empty or contains no valid data.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
                return jsonify({'success': False, 'error': error_msg}), 400
            flash(error_msg, 'error')
            return redirect(url_for('upload_page'))
        
        # Generate analysis
        analysis = generate_data_analysis(df, timestamp)
        
        # Get AI insights
        ai_insights = get_ai_insights(df, analysis)
        
        # Prepare basic stats for template compatibility
        basic_stats = {
            'rows': analysis['basic_info']['shape'][0],
            'columns': analysis['basic_info']['shape'][1],
            'shape': analysis['basic_info']['shape'],
            'numeric_columns': [col for col, dtype in analysis['basic_info']['dtypes'].items() if 'int' in str(dtype) or 'float' in str(dtype)],
            'missing_percentage': analysis['data_quality']['missing_percentage'],
            'memory_mb': round(analysis['basic_info']['memory_usage'] / (1024 * 1024), 2),
            'data_types': list(set(analysis['basic_info']['dtypes'].values()))
        }
        
        # Generate visualizations (plots are already generated in analysis)
        plots = analysis.get('plots', {})
        
        # Store results in session for chat functionality - with careful type conversion
        try:
            session_data = {
                'basic_info': {
                    'shape': list(analysis['basic_info']['shape']),  # Convert tuple to list
                    'columns': list(analysis['basic_info']['columns']),
                    'dtypes': {str(k): str(v) for k, v in analysis['basic_info']['dtypes'].items()},
                    'memory_usage': int(analysis['basic_info']['memory_usage'])
                },
                'columns_info': {
                    'column_names': list(analysis['basic_info']['columns']),
                    'numeric_columns': [str(col) for col, dtype in analysis['basic_info']['dtypes'].items() if 'int' in str(dtype) or 'float' in str(dtype)],
                    'categorical_columns': [str(col) for col, dtype in analysis['basic_info']['dtypes'].items() if 'int' not in str(dtype) and 'float' not in str(dtype)]
                },
                'missing_data': {
                    'total_missing': int(analysis['data_quality']['total_missing']),
                    'missing_percentage': float(analysis['data_quality']['missing_percentage']),
                    'missing_by_column': {str(k): float(v) for k, v in analysis['missing_data']['missing_percentages'].items()}
                },
                'statistics': convert_numpy_types(analysis.get('statistics', {})),
                'correlation_analysis': convert_numpy_types(analysis.get('correlation_analysis', {})),
                'outlier_analysis': convert_numpy_types(analysis.get('outlier_analysis', {})),
                'distribution_analysis': convert_numpy_types(analysis.get('distribution_analysis', {})),
                'data_quality': convert_numpy_types(analysis.get('data_quality', {})),
                'filename': str(filename),
                'timestamp': str(timestamp)
            }
            
            # Convert all numpy types to native Python types
            converted_data = convert_numpy_types(session_data)
            
            # Test serialization before storing
            import json
            try:
                json.dumps(converted_data)  # Test if it's serializable
                session['analysis_results'] = converted_data
            except (TypeError, ValueError) as e:
                logger.error(f"Session data not serializable: {e}")
                # Store minimal safe data
                session['analysis_results'] = {
                    'basic_info': {
                        'shape': [len(df), len(df.columns)],
                        'columns': [str(col) for col in df.columns],
                        'dtypes': {str(col): str(df[col].dtype) for col in df.columns},
                        'memory_usage': 0
                    },
                    'columns_info': {
                        'column_names': [str(col) for col in df.columns],
                        'numeric_columns': [str(col) for col in df.select_dtypes(include=[np.number]).columns],
                        'categorical_columns': [str(col) for col in df.select_dtypes(exclude=[np.number]).columns]
                    },
                    'missing_data': {
                        'total_missing': int(df.isnull().sum().sum()),
                        'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
                        'missing_by_column': {str(col): float((df[col].isnull().sum() / len(df)) * 100) for col in df.columns}
                    },
                    'filename': str(filename),
                    'timestamp': str(timestamp)
                }
            
        except Exception as e:
            logger.error(f"Error storing session data: {e}")
            # Store minimal data if conversion fails
            session['analysis_results'] = {
                'basic_info': {
                    'shape': [0, 0],
                    'columns': [],
                    'dtypes': {},
                    'memory_usage': 0
                },
                'columns_info': {
                    'column_names': [],
                    'numeric_columns': [],
                    'categorical_columns': []
                },
                'missing_data': {
                    'total_missing': 0,
                    'missing_percentage': 0.0,
                    'missing_by_column': {}
                },
                'filename': str(filename),
                'timestamp': str(timestamp)
            }
        
        # Store the actual dataframe data for more detailed analysis
        try:
            # Convert dataframe sample to safe format
            df_sample_raw = df.head(20).to_dict('records')
            df_sample_converted = convert_numpy_types(df_sample_raw)
            
            # Test serialization
            import json
            json.dumps(df_sample_converted)  # Test if serializable
            session['df_sample'] = df_sample_converted
            
            # Convert describe data to safe format
            if not df.select_dtypes(include=[np.number]).empty:
                df_describe_raw = df.describe().to_dict()
                df_describe_converted = convert_numpy_types(df_describe_raw)
                json.dumps(df_describe_converted)  # Test if serializable
                session['df_describe'] = df_describe_converted
            else:
                session['df_describe'] = {}
                
        except Exception as e:
            logger.warning(f"Could not store dataframe data in session: {e}")
            session['df_sample'] = []
            session['df_describe'] = {}
        
        # Prepare results for template
        results = {
            'filename': filename,
            'timestamp': timestamp,
            'analysis': analysis,
            'ai_insights': ai_insights,
            'basic_stats': basic_stats,
            'plots': plots,
            'sample_data': df.head(10).to_html(classes='table table-striped', escape=False)
        }
        
        try:
            if db and db.connected:
                analysis_to_save = convert_numpy_types(analysis)
                
                # Prepare sample data for MongoDB storage
                df_sample_data = []
                df_describe_data = {}
                try:
                    df_sample_raw = df.head(20).to_dict('records')
                    df_sample_data = convert_numpy_types(df_sample_raw)
                except Exception as e:
                    logger.warning(f"Could not prepare dataframe sample: {e}")
                
                try:
                    if not df.select_dtypes(include=[np.number]).empty:
                        df_describe_raw = df.describe().to_dict()
                        df_describe_data = convert_numpy_types(df_describe_raw)
                except Exception as e:
                    logger.warning(f"Could not prepare dataframe describe: {e}")
                
                comprehensive_data = {
                    'basic_info': {
                        'shape': list(analysis['basic_info']['shape']),
                        'columns': list(analysis['basic_info']['columns']),
                        'dtypes': {str(k): str(v) for k, v in analysis['basic_info']['dtypes'].items()},
                        'memory_usage': int(analysis['basic_info']['memory_usage'])
                    },
                    'columns_info': {
                        'column_names': list(analysis['basic_info']['columns']),
                        'numeric_columns': [str(col) for col, dtype in analysis['basic_info']['dtypes'].items() if 'int' in str(dtype) or 'float' in str(dtype)],
                        'categorical_columns': [str(col) for col, dtype in analysis['basic_info']['dtypes'].items() if 'int' not in str(dtype) and 'float' not in str(dtype)]
                    },
                    'missing_data': {
                        'total_missing': int(analysis['data_quality']['total_missing']),
                        'missing_percentage': float(analysis['data_quality']['missing_percentage']),
                        'missing_by_column': {str(k): float(v) for k, v in analysis['missing_data']['missing_percentages'].items()}
                    },
                    'statistics': convert_numpy_types(analysis.get('statistics', {})),
                    'correlation_analysis': convert_numpy_types(analysis.get('correlation_analysis', {})),
                    'outlier_analysis': convert_numpy_types(analysis.get('outlier_analysis', {})),
                    'distribution_analysis': convert_numpy_types(analysis.get('distribution_analysis', {})),
                    'data_quality': convert_numpy_types(analysis.get('data_quality', {})),
                    'df_sample': df_sample_data,
                    'df_describe': df_describe_data,
                    'full_analysis': analysis_to_save,
                    'filename': str(filename),
                    'timestamp': str(timestamp)
                }
                
                db.save_analysis(timestamp, comprehensive_data)
                
                for plot_type, plot_filename in plots.items():
                    plot_metadata = {
                        'plot_filename': plot_filename,
                        'plot_type': plot_type,
                        'filename': filename,
                        'shape': analysis['basic_info']['shape'],
                        'columns': analysis['basic_info']['columns']
                    }
                    db.save_plot(timestamp, plot_metadata)
        except Exception as e:
            logger.warning(f"Failed to save analysis to MongoDB: {e}")
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
            return jsonify({'success': True, 'redirect': '/results'})
        
        return render_template('results_dashboard.html', **results)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        
        # Provide user-friendly error messages
        error_str = str(e).lower()
        if "gaussian_kde" in error_str or "singular" in error_str:
            error_msg = "⚠️ Your data has some columns with identical values that prevent advanced statistical analysis. The basic analysis will still work, but some visualizations may be limited. Try using data with more varied values."
        elif "memory" in error_str:
            error_msg = "📁 Your file is too large to process. Please try a smaller file (under 50MB)."
        elif "parse" in error_str or "read" in error_str:
            error_msg = "📄 Unable to read your file. Please ensure it's a valid CSV or Excel file with proper formatting."
        else:
            error_msg = f"❌ An error occurred: {str(e)}"
        
        # Check if this is an AJAX request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/json':
            return jsonify({'success': False, 'error': error_msg}), 400
        
        flash(error_msg, 'error')
        return redirect(url_for('upload_page'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/presentation')
def presentation():
    """Project presentation page"""
    return render_template('presentation.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/test_grid.html')
def test_grid():
    """Test page for grid background"""
    return send_from_directory('.', 'test_grid.html')

@app.route('/test_upload.html')
def test_upload():
    """Test page for upload functionality"""
    return send_from_directory('.', 'test_upload.html')



@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Internal error: {error}")
    return render_template('500.html'), 500

@app.route('/export_data/<timestamp>')
def export_data(timestamp):
    """Export cleaned data as CSV"""
    try:
        return jsonify({
            'status': 'success',
            'message': 'Data export feature - coming soon',
            'timestamp': timestamp
        })
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    """Chat with Gemini AI with data context"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Configure Gemini AI
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return jsonify({'error': 'Gemini API key not configured'}), 500
            
        genai.configure(api_key=api_key)
        use_phi3 = os.environ.get('USE_PHI3_MODEL', 'False') == 'True'
        
        ai_response = None
        
        if use_phi3 and generate_chat_response:
            try:
                ai_response = generate_chat_response(user_message)
            except Exception as phi3_err:
                logger.warning(f"Phi-3 response generation failed: {phi3_err}, falling back to Gemini")
                ai_response = None
        
        if not ai_response and model:
            try:
                response = model.generate_content(user_message)
                ai_response = response.text if response and response.text else None
            except Exception as gemini_err:
                logger.warning(f"Gemini response generation failed: {gemini_err}")
                ai_response = None
        
        # Get analysis data from session - with error handling
        try:
            analysis_data = session.get('analysis_results')
        except Exception as e:
            logger.error(f"Session access error: {e}")
            analysis_data = None
        
        # If session data not available, try to retrieve from MongoDB
        if not analysis_data and db and db.connected:
            try:
                mongo_analysis = db.get_latest_analysis()
                if mongo_analysis:
                    analysis_data = mongo_analysis
                    logger.info("Retrieved analysis data from MongoDB for chat context")
            except Exception as e:
                logger.warning(f"Failed to retrieve analysis from MongoDB: {e}")
        
        data_context = ""
        
        # Check if we have valid analysis data with actual content
        has_valid_data = (analysis_data and 
                         isinstance(analysis_data, dict) and 
                         analysis_data.get('basic_info', {}).get('shape'))
        
        logger.debug(f"Analysis data status: available={bool(analysis_data)}, valid={has_valid_data}")
        if analysis_data:
            logger.debug(f"Analysis data keys: {list(analysis_data.keys())}")
        
        if has_valid_data:
            # Create comprehensive data context
            basic_info = analysis_data.get('basic_info', {})
            columns_info = analysis_data.get('columns_info', {})
            missing_data = analysis_data.get('missing_data', {})
            statistics = analysis_data.get('statistics', {})
            correlation_analysis = analysis_data.get('correlation_analysis', {})
            outlier_analysis = analysis_data.get('outlier_analysis', {})
            distribution_analysis = analysis_data.get('distribution_analysis', {})
            
            # Get sample data for more context (try session first, then MongoDB)
            df_sample = session.get('df_sample', [])
            df_describe = session.get('df_describe', {})
            
            # If not in session, get from analysis_data
            if not df_sample and analysis_data:
                df_sample = analysis_data.get('df_sample', [])
            if not df_describe and analysis_data:
                df_describe = analysis_data.get('df_describe', {})
            
            # Create detailed missing data info
            missing_columns = []
            for col, pct in missing_data.get('missing_by_column', {}).items():
                if pct > 0:
                    missing_columns.append(f"{col}: {pct:.1f}%")
            
            # Create correlation insights
            correlation_insights = []
            if correlation_analysis.get('high_correlations'):
                for corr in correlation_analysis['high_correlations'][:3]:
                    correlation_insights.append(f"{corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.2f}")
            
            # Create outlier insights
            outlier_insights = []
            for col, outlier_info in outlier_analysis.items():
                if isinstance(outlier_info, dict) and outlier_info.get('count', 0) > 0:
                    outlier_insights.append(f"{col}: {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)")
            
            data_context = f"""
📊 CURRENT DATASET CONTEXT:
=========================
Dataset Overview:
• File: {analysis_data.get('filename', 'Unknown')}
• Shape: {basic_info.get('shape', 'Unknown')} (rows × columns)
• Memory Usage: {basic_info.get('memory_usage', 0) / (1024*1024):.1f} MB

Column Information:
• Total Columns: {len(columns_info.get('column_names', []))}
• All Columns: {', '.join(columns_info.get('column_names', []))}
• Numeric Columns ({len(columns_info.get('numeric_columns', []))}): {', '.join(columns_info.get('numeric_columns', []))}
• Categorical Columns ({len(columns_info.get('categorical_columns', []))}): {', '.join(columns_info.get('categorical_columns', []))}

Data Quality:
• Missing Values: {missing_data.get('total_missing', 0)} total ({missing_data.get('missing_percentage', 0):.1f}%)
• Columns with Missing Data: {'; '.join(missing_columns[:5])}

Statistical Summary:
{str(df_describe)[:500] if df_describe else 'No numeric data available'}

Correlations Found:
{'; '.join(correlation_insights) if correlation_insights else 'No strong correlations detected'}

Outliers Detected:
{'; '.join(outlier_insights[:5]) if outlier_insights else 'No significant outliers detected'}

Sample Data (first few rows):
{str(df_sample[:3]) if df_sample else 'No sample data available'}
"""
        
        # Create enhanced system prompt with data context
        system_prompt = f"""You are an expert AI data analysis assistant with FULL ACCESS to the user's uploaded dataset. 

{data_context}

CRITICAL INSTRUCTIONS:
• ALWAYS reference the ACTUAL dataset shown above - never give generic responses
• Use SPECIFIC column names, values, and statistics from the data context
• When discussing missing data, mention the EXACT columns and percentages shown above
• When discussing correlations, reference the SPECIFIC correlations found
• When discussing outliers, mention the EXACT columns and counts shown above
• Provide CONCRETE, actionable advice based on THIS SPECIFIC dataset
• If asked about patterns, refer to the actual data patterns visible in the sample data
• Be conversational and helpful, using emojis appropriately
• Always start responses by acknowledging the specific dataset (mention filename if available)
• If the user asks about something not visible in the data context, say so explicitly

RESPONSE STYLE:
• Start with: "Looking at your [filename] dataset..." or "Based on your data with [X] rows and [Y] columns..."
• Use bullet points for clarity
• Include specific numbers and percentages from the actual data
• Suggest next steps based on what you can see in the data

Remember: You have the ACTUAL data context above - use it! Never give generic responses."""
        
        # Handle case when no data is available
        if not has_valid_data:
            ai_response = """🤖 Hi there! I'd love to help you analyze your data, but I don't currently have access to any uploaded dataset. 

To get started:
1. 📁 Upload a CSV or Excel file using the upload page
2. 📊 Wait for the analysis to complete
3. 💬 Then come back here and ask me anything about your data!

Once you've uploaded data, I'll be able to answer specific questions about:
• Missing data patterns and recommendations
• Correlations between variables
• Outlier detection and handling
• Statistical insights and trends
• Data quality assessment
• Visualization suggestions

Looking forward to analyzing your data! 🚀"""
            
            return jsonify({
                'response': ai_response,
                'status': 'success'
            })
        
        full_prompt = f"{system_prompt}\n\nUser Question: {user_message}\n\nResponse:"
        
        # Generate response
        if not (use_phi3 and generate_chat_response and ai_response):
            response = model.generate_content(full_prompt)
            ai_response = response.text if response.text else "I'm here to help analyze your data! Ask me anything about your dataset."
        
        return jsonify({
            'response': ai_response,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({
            'response': "I apologize, but I'm having trouble processing your request right now. Please try again!",
            'status': 'error'
        }), 500

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'name': 'DataScope',
        'ai_enabled': model is not None
    })

@app.route('/contact', methods=['POST'])
def submit_contact():
    """Handle contact form submissions"""
    try:
        # Get form data
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        subject = request.form.get('subject', '').strip()
        message = request.form.get('message', '').strip()
        
        # Validate required fields
        if not all([name, email, subject, message]):
            return jsonify({
                'success': False,
                'message': 'Please fill in all required fields.'
            }), 400
        
        # Create email content
        email_subject = f"Contact Form: {subject} - From {name}"
        email_body = f"""
New contact form submission from DataScope:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
Sent from DataScope Contact Form
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # Try to send email (using a simple approach)
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = 'mangaldipdhua@gmail.com'
            msg['Subject'] = email_subject
            
            # Add body to email
            msg.attach(MIMEText(email_body, 'plain'))
            
            # Log the message for debugging
            logger.info(f"Contact form submission received:")
            logger.info(f"From: {name} ({email})")
            logger.info(f"Subject: {subject}")
            logger.info(f"Message: {message}")
            
            # Try to send email if SMTP is configured
            smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.environ.get('SMTP_PORT', '587'))
            smtp_username = os.environ.get('SMTP_USERNAME', '')
            smtp_password = os.environ.get('SMTP_PASSWORD', '')
            
            # Debug logging
            logger.info(f"SMTP Config - Server: {smtp_server}, Port: {smtp_port}")
            logger.info(f"SMTP Username: {'***' if smtp_username else 'NOT SET'}")
            logger.info(f"SMTP Password: {'***' if smtp_password else 'NOT SET'}")
            
            if smtp_username and smtp_password:
                try:
                    server = smtplib.SMTP(smtp_server, smtp_port)
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    
                    # Create a proper message that appears to come from the sender
                    msg = MIMEMultipart()
                    msg['From'] = f"{name} <{email}>"  # Show sender's name and email
                    msg['To'] = 'mangaldipdhua@gmail.com'
                    msg['Subject'] = email_subject
                    msg['Reply-To'] = email  # Easy reply to sender
                    
                    # Enhanced email body with better formatting
                    enhanced_body = f"""
Contact Form Submission - DataScope

From: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
This message was sent via the DataScope contact form.
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
IP: {request.environ.get('REMOTE_ADDR', 'Unknown')}

You can reply directly to this email to respond to {name}.
                    """
                    
                    # Add body
                    msg.attach(MIMEText(enhanced_body, 'plain'))
                    
                    # Send notification email to you
                    text = msg.as_string()
                    server.sendmail(smtp_username, 'mangaldipdhua@gmail.com', text)
                    
                    # Send automatic reply to the user
                    reply_msg = MIMEMultipart()
                    reply_msg['From'] = f"Mangaldip Dhua - DataScope <{smtp_username}>"
                    reply_msg['To'] = email
                    reply_msg['Subject'] = f"Thank you for contacting DataScope - {subject}"
                    
                    # Create professional auto-reply
                    reply_body = f"""
Dear {name},

Thank you for reaching out to DataScope! I have received your message regarding "{subject}" and truly appreciate you taking the time to contact me.

Your Message:
"{message}"

I will review your inquiry and get back to you as soon as possible, typically within 24-48 hours. In the meantime, feel free to explore DataScope's features and capabilities.

If you have any urgent questions or need immediate assistance, you can also reach me directly at:
• Email: mangaldipdhua@gmail.com
• LinkedIn: www.linkedin.com/in/mangaldipdhua
• GitHub: https://github.com/mangaldipdhua

Thank you for your interest in DataScope. I look forward to connecting with you soon!

Best regards,
Mangaldip Dhua
AI/ML Engineer & Creator of DataScope

---
This is an automated response. Please do not reply to this email directly.
For urgent matters, contact: mangaldipdhua@gmail.com

DataScope - Making AI-powered data analysis accessible to everyone
Website: {request.host_url}
                    """
                    
                    reply_msg.attach(MIMEText(reply_body, 'plain'))
                    
                    # Send auto-reply
                    reply_text = reply_msg.as_string()
                    server.sendmail(smtp_username, email, reply_text)
                    
                    server.quit()
                    
                    logger.info(f"Email sent successfully from {name} ({email}) to mangaldipdhua@gmail.com")
                    logger.info(f"Auto-reply sent to {name} ({email})")
                except Exception as email_error:
                    logger.error(f"Failed to send email: {str(email_error)}")
            else:
                logger.info("SMTP not configured - message logged only")
                
                # Save to file as backup
                try:
                    contact_file = Path('contact_messages.txt')
                    with open(contact_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*50}\n")
                        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Name: {name}\n")
                        f.write(f"Email: {email}\n")
                        f.write(f"Subject: {subject}\n")
                        f.write(f"Message:\n{message}\n")
                        f.write(f"{'='*50}\n")
                    logger.info("Contact message saved to contact_messages.txt")
                except Exception as file_error:
                    logger.error(f"Failed to save contact message to file: {str(file_error)}")
            
            return jsonify({
                'success': True,
                'message': 'Thank you for your message! We\'ll get back to you within 24 hours.'
            })
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            # Still return success since we logged the message
            return jsonify({
                'success': True,
                'message': 'Thank you for your message! We\'ll get back to you within 24 hours.'
            })
            
    except Exception as e:
        logger.error(f"Error processing contact form: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Sorry, there was an error processing your message. Please try again.'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
    
    
    
    
    
    
    
    
    
    