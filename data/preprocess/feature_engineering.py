"""
Feature Engineering Pipeline for KubeShield
Combines features from LID-DS, CICIDS2018, and audit logs
with normalization and meta-feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline"""
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    enable_meta_features: bool = True
    enable_tfidf_features: bool = True
    window_size: int = 10
    quantile_features: List[float] = None
    
    def __post_init__(self):
        if self.quantile_features is None:
            self.quantile_features = [0.25, 0.5, 0.75]


class StatisticalFeatureEngineer:
    """Extract statistical meta-features from numeric data"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        logger.info(f"Initialized StatisticalFeatureEngineer with window_size={window_size}")
    
    def extract_statistical_features(
        self, 
        data: pd.DataFrame, 
        numeric_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extract statistical features including mean, std, min, max, skewness
        
        Args:
            data: Input dataframe
            numeric_cols: List of numeric column names
            
        Returns:
            Dataframe with added statistical features
        """
        df = data.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            # Rolling statistics
            df[f'{col}_rolling_mean'] = df[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).std().fillna(0)
            
            # Cumulative statistics
            df[f'{col}_cumsum'] = df[col].cumsum()
            df[f'{col}_cumprod'] = df[col].cumprod()
            
            # Difference features
            df[f'{col}_diff'] = df[col].diff().fillna(0)
            df[f'{col}_pct_change'] = df[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            
            # Distribution features
            df[f'{col}_skewness'] = df[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).apply(lambda x: x.skew(), raw=False).fillna(0)
            
            df[f'{col}_kurtosis'] = df[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).apply(lambda x: x.kurtosis(), raw=False).fillna(0)
        
        logger.info(f"Extracted statistical features for {len(numeric_cols)} columns")
        return df
    
    def extract_quantile_features(
        self, 
        data: pd.DataFrame, 
        numeric_cols: List[str],
        quantiles: List[float] = None
    ) -> pd.DataFrame:
        """Extract quantile-based features"""
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        
        df = data.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            for q in quantiles:
                df[f'{col}_quantile_{int(q*100)}'] = df[col].rolling(
                    window=self.window_size,
                    min_periods=1
                ).quantile(q).values
        
        logger.info(f"Extracted quantile features with quantiles: {quantiles}")
        return df


class NetworkFlowFeatureEngineer:
    """Extract features specific to network flow data (CICIDS2018, LID-DS)"""
    
    @staticmethod
    def extract_traffic_pattern_features(data: pd.DataFrame) -> pd.DataFrame:
        """Extract traffic pattern features"""
        df = data.copy()
        
        # Flow duration statistics
        if 'Flow Duration' in df.columns or 'flow_duration' in df.columns:
            col = 'Flow Duration' if 'Flow Duration' in df.columns else 'flow_duration'
            df['flow_duration_log'] = np.log1p(df[col])
            df['flow_duration_squared'] = df[col] ** 2
        
        # Packet and byte statistics
        packet_cols = [col for col in df.columns if 'packet' in col.lower()]
        byte_cols = [col for col in df.columns if 'byte' in col.lower()]
        
        if packet_cols:
            df['total_packets'] = df[packet_cols].sum(axis=1)
            df['avg_packet_size'] = np.divide(
                df[[col for col in byte_cols if col in df.columns]].sum(axis=1),
                df['total_packets'],
                where=df['total_packets'] != 0,
                out=np.zeros_like(df['total_packets'])
            )
        
        # Protocol distribution
        protocol_cols = [col for col in df.columns if 'protocol' in col.lower()]
        if protocol_cols:
            for col in protocol_cols:
                if df[col].dtype == 'object':
                    # One-hot encode protocols
                    protocol_dummies = pd.get_dummies(df[col], prefix='protocol')
                    df = pd.concat([df, protocol_dummies], axis=1)
        
        logger.info("Extracted traffic pattern features")
        return df
    
    @staticmethod
    def extract_flow_rate_features(data: pd.DataFrame) -> pd.DataFrame:
        """Extract flow rate and velocity features"""
        df = data.copy()
        
        # Flow rate calculations
        flow_rate_cols = [col for col in df.columns if 'rate' in col.lower()]
        
        if flow_rate_cols:
            for col in flow_rate_cols:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(np.abs(df[col]))
                df[f'{col}_acceleration'] = df[col].diff().fillna(0)
        
        logger.info("Extracted flow rate features")
        return df
    
    @staticmethod
    def extract_bidirectional_features(data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from bidirectional flow data"""
        df = data.copy()
        
        # Forward and backward statistics
        forward_cols = [col for col in df.columns if 'forward' in col.lower() or 'fwd' in col.lower()]
        backward_cols = [col for col in df.columns if 'backward' in col.lower() or 'bwd' in col.lower()]
        
        numeric_forward = [col for col in forward_cols if df[col].dtype in [np.float64, np.int64]]
        numeric_backward = [col for col in backward_cols if df[col].dtype in [np.float64, np.int64]]
        
        if numeric_forward and numeric_backward:
            fwd_sum = df[numeric_forward].sum(axis=1)
            bwd_sum = df[numeric_backward].sum(axis=1)
            
            # Ratios
            df['forward_backward_ratio'] = np.divide(
                fwd_sum, bwd_sum,
                where=bwd_sum != 0,
                out=np.zeros_like(fwd_sum)
            )
            df['forward_backward_diff'] = fwd_sum - bwd_sum
            
            # Entropy-like measures
            total = fwd_sum + bwd_sum
            df['directional_entropy'] = np.where(
                total > 0,
                -(fwd_sum/total * np.log2(fwd_sum/total + 1e-10) + 
                  bwd_sum/total * np.log2(bwd_sum/total + 1e-10)),
                0
            )
        
        logger.info("Extracted bidirectional flow features")
        return df


class AuditLogFeatureEngineer:
    """Extract features from Kubernetes audit logs"""
    
    @staticmethod
    def extract_audit_event_features(data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from audit log events"""
        df = data.copy()
        
        # Event type features
        if 'verb' in df.columns:
            df['is_get_request'] = (df['verb'] == 'get').astype(int)
            df['is_create_request'] = (df['verb'] == 'create').astype(int)
            df['is_delete_request'] = (df['verb'] == 'delete').astype(int)
            df['is_update_request'] = (df['verb'] == 'update').astype(int)
            df['is_watch_request'] = (df['verb'] == 'watch').astype(int)
        
        # Status code features
        if 'code' in df.columns:
            df['is_error_status'] = (df['code'] >= 400).astype(int)
            df['is_success_status'] = (df['code'] < 400).astype(int)
            df['is_forbidden'] = (df['code'] == 403).astype(int)
            df['is_unauthorized'] = (df['code'] == 401).astype(int)
        
        # User/Source features
        if 'user' in df.columns:
            user_counts = df['user'].value_counts()
            df['user_request_frequency'] = df['user'].map(user_counts)
            
            # User privilege level estimation
            df['is_system_user'] = df['user'].str.contains(
                'system|admin|root', 
                case=False, 
                na=False
            ).astype(int)
        
        if 'sourceIPs' in df.columns or 'source_ip' in df.columns:
            src_col = 'sourceIPs' if 'sourceIPs' in df.columns else 'source_ip'
            ip_counts = df[src_col].value_counts()
            df['source_ip_frequency'] = df[src_col].map(ip_counts)
        
        logger.info("Extracted audit event features")
        return df
    
    @staticmethod
    def extract_audit_sequence_features(
        data: pd.DataFrame, 
        groupby_col: str = 'user',
        window_size: int = 10
    ) -> pd.DataFrame:
        """Extract sequential patterns from audit logs"""
        df = data.copy()
        
        if groupby_col not in df.columns:
            logger.warning(f"Column {groupby_col} not found for sequence features")
            return df
        
        # Requests per user
        df['user_request_count'] = df.groupby(groupby_col).cumcount() + 1
        
        # Time-based sequence features
        if 'timestamp' in df.columns or 'time' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df['time_delta'] = df.groupby(groupby_col)[time_col].diff().dt.total_seconds().fillna(0)
                df['avg_request_interval'] = df.groupby(groupby_col)['time_delta'].transform('mean').fillna(0)
            except Exception as e:
                logger.warning(f"Could not extract time-based features: {e}")
        
        logger.info("Extracted audit sequence features")
        return df
    
    @staticmethod
    def extract_tfidf_features(
        data: pd.DataFrame, 
        text_col: str = 'request_object',
        max_features: int = 50
    ) -> Tuple[pd.DataFrame, TfidfVectorizer]:
        """Extract TF-IDF features from text fields"""
        if text_col not in data.columns:
            logger.warning(f"Column {text_col} not found for TF-IDF features")
            return data, None
        
        try:
            vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(data[text_col].fillna(''))
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'tfidf_{name}' for name in vectorizer.get_feature_names_out()]
            )
            
            df = pd.concat([data, tfidf_df], axis=1)
            logger.info(f"Extracted {max_features} TF-IDF features")
            return df, vectorizer
        except Exception as e:
            logger.warning(f"Could not extract TF-IDF features: {e}")
            return data, None


class FeatureNormalizer:
    """Normalize and scale features"""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer
        
        Args:
            method: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), or 'robust' (RobustScaler)
        """
        self.method = method
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.fitted = False
        logger.info(f"Initialized FeatureNormalizer with method={method}")
    
    def fit(self, data: pd.DataFrame, numeric_cols: List[str] = None) -> 'FeatureNormalizer':
        """Fit the scaler on training data"""
        if numeric_cols is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.numeric_cols = numeric_cols
        self.scaler.fit(data[numeric_cols])
        self.fitted = True
        logger.info(f"Fitted normalizer on {len(numeric_cols)} numeric columns")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        df = data.copy()
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        logger.info(f"Transformed data with shape {df.shape}")
        return df
    
    def fit_transform(self, data: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        if numeric_cols is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.fit(data, numeric_cols)
        return self.transform(data)


class FeatureEngineeringPipeline:
    """Main feature engineering pipeline combining all components"""
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """
        Initialize the pipeline
        
        Args:
            config: FeatureEngineeringConfig object
        """
        self.config = config or FeatureEngineeringConfig()
        self.stat_engineer = StatisticalFeatureEngineer(window_size=self.config.window_size)
        self.network_engineer = NetworkFlowFeatureEngineer()
        self.audit_engineer = AuditLogFeatureEngineer()
        self.normalizer = FeatureNormalizer(method=self.config.normalization_method)
        self.tfidf_vectorizer = None
        logger.info("Initialized FeatureEngineeringPipeline")
    
    def process_cicids2018(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process CICIDS2018 dataset"""
        logger.info("Processing CICIDS2018 dataset")
        df = data.copy()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna(thresh=len(df) * 0.5, axis=1)
        
        # Extract network flow features
        df = self.network_engineer.extract_traffic_pattern_features(df)
        df = self.network_engineer.extract_flow_rate_features(df)
        df = self.network_engineer.extract_bidirectional_features(df)
        
        # Extract statistical meta-features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.config.enable_meta_features:
            df = self.stat_engineer.extract_statistical_features(df, numeric_cols)
            df = self.stat_engineer.extract_quantile_features(
                df, 
                numeric_cols,
                self.config.quantile_features
            )
        
        logger.info(f"CICIDS2018 processing complete. Shape: {df.shape}")
        return df
    
    def process_lids_ds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process LID-DS dataset"""
        logger.info("Processing LID-DS dataset")
        df = data.copy()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna(thresh=len(df) * 0.5, axis=1)
        
        # Extract network flow features
        df = self.network_engineer.extract_traffic_pattern_features(df)
        df = self.network_engineer.extract_flow_rate_features(df)
        df = self.network_engineer.extract_bidirectional_features(df)
        
        # Extract statistical meta-features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.config.enable_meta_features:
            df = self.stat_engineer.extract_statistical_features(df, numeric_cols)
            df = self.stat_engineer.extract_quantile_features(
                df, 
                numeric_cols,
                self.config.quantile_features
            )
        
        logger.info(f"LID-DS processing complete. Shape: {df.shape}")
        return df
    
    def process_audit_logs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process Kubernetes audit logs"""
        logger.info("Processing Kubernetes audit logs")
        df = data.copy()
        
        # Extract audit-specific features
        df = self.audit_engineer.extract_audit_event_features(df)
        df = self.audit_engineer.extract_audit_sequence_features(df)
        
        # Extract TF-IDF features
        if self.config.enable_tfidf_features:
            df, self.tfidf_vectorizer = self.audit_engineer.extract_tfidf_features(df)
        
        logger.info(f"Audit logs processing complete. Shape: {df.shape}")
        return df
    
    def combine_features(
        self, 
        cicids_data: Optional[pd.DataFrame] = None,
        lids_data: Optional[pd.DataFrame] = None,
        audit_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Combine features from multiple sources
        
        Args:
            cicids_data: Processed CICIDS2018 data
            lids_data: Processed LID-DS data
            audit_data: Processed audit log data
            
        Returns:
            Combined feature dataframe
        """
        dataframes = []
        
        if cicids_data is not None:
            dataframes.append(self.process_cicids2018(cicids_data))
        
        if lids_data is not None:
            dataframes.append(self.process_lids_ds(lids_data))
        
        if audit_data is not None:
            dataframes.append(self.process_audit_logs(audit_data))
        
        if not dataframes:
            raise ValueError("At least one dataset must be provided")
        
        # Combine horizontally
        combined = pd.concat(dataframes, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        logger.info(f"Combined features from {len(dataframes)} sources. Shape: {combined.shape}")
        return combined
    
    def normalize_features(
        self, 
        data: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize numeric features
        
        Args:
            data: Input dataframe
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized dataframe
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if fit:
            normalized_data = self.normalizer.fit_transform(data, numeric_cols)
        else:
            normalized_data = self.normalizer.transform(data)
        
        # Preserve categorical columns
        for col in categorical_cols:
            if col in data.columns:
                normalized_data[col] = data[col]
        
        logger.info(f"Normalized {len(numeric_cols)} numeric features")
        return normalized_data
    
    def execute_pipeline(
        self,
        cicids_data: Optional[pd.DataFrame] = None,
        lids_data: Optional[pd.DataFrame] = None,
        audit_data: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute complete feature engineering pipeline
        
        Args:
            cicids_data: CICIDS2018 dataframe
            lids_data: LID-DS dataframe
            audit_data: Audit logs dataframe
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (processed dataframe, metadata dictionary)
        """
        logger.info("Starting feature engineering pipeline execution")
        
        # Combine features
        combined = self.combine_features(cicids_data, lids_data, audit_data)
        
        # Normalize if requested
        if normalize:
            combined = self.normalize_features(combined, fit=True)
        
        # Generate metadata
        metadata = {
            'num_samples': len(combined),
            'num_features': combined.shape[1],
            'feature_names': combined.columns.tolist(),
            'numeric_features': combined.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': combined.select_dtypes(include=['object']).columns.tolist(),
            'config': self.config.__dict__,
            'has_tfidf': self.tfidf_vectorizer is not None,
        }
        
        logger.info(f"Pipeline execution complete. Output shape: {combined.shape}")
        logger.info(f"Features: {metadata['num_features']}")
        
        return combined, metadata


# Example usage and testing
if __name__ == "__main__":
    # Create sample configuration
    config = FeatureEngineeringConfig(
        normalization_method='standard',
        enable_meta_features=True,
        enable_tfidf_features=True,
        window_size=10,
        quantile_features=[0.25, 0.5, 0.75]
    )
    
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(config)
    
    logger.info("Feature engineering module loaded successfully")
