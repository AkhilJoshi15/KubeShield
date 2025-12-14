// KubeShield - Example Flink Job for Feature Engineering
// Processes raw telemetry from Kafka, generates feature vectors for ML models

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaConsumer;
import org.apache.flink.streaming.connectors.kafka.FlinkKafkaProducer;
import org.json.JSONObject;
import java.util.*;
import java.util.stream.Collectors;

public class FlinkJob {
    
    /**
     * Represents a raw telemetry event from Kafka
     */
    public static class TelemetryEvent {
        public long timestamp;
        public String nodeName;
        public String userName;
        public String action;
        public double cpuUsage;
        public double memoryUsage;
        public int processCount;
        
        public TelemetryEvent(JSONObject json) {
            this.timestamp = json.optLong("timestamp", System.currentTimeMillis());
            this.nodeName = json.optString("node_name", "unknown");
            this.userName = json.optString("user_name", "unknown");
            this.action = json.optString("action", "unknown");
            this.cpuUsage = json.optDouble("cpu_usage", 0.0);
            this.memoryUsage = json.optDouble("memory_usage", 0.0);
            this.processCount = json.optInt("process_count", 0);
        }
    }
    
    /**
     * Represents extracted feature vector for ML models
     */
    public static class FeatureVector {
        public long timestamp;
        public String nodeName;
        public double[] features; // 73 features as per paper
        
        public FeatureVector(long timestamp, String nodeName, double[] features) {
            this.timestamp = timestamp;
            this.nodeName = nodeName;
            this.features = features;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"timestamp\":").append(timestamp).append(",");
            sb.append("\"node_name\":\"").append(nodeName).append("\",");
            sb.append("\"features\":[");
            for (int i = 0; i < features.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(features[i]);
            }
            sb.append("]}");
            return sb.toString();
        }
    }
    
    /**
     * Feature extraction map function
     * Converts raw telemetry to feature vectors
     */
    public static class FeatureExtractionMapper implements MapFunction<String, FeatureVector> {
        
        @Override
        public FeatureVector map(String value) throws Exception {
            JSONObject json = new JSONObject(value);
            TelemetryEvent event = new TelemetryEvent(json);
            
            // Extract 73 features (simplified version)
            double[] features = new double[73];
            
            // Basic statistical features
            features[0] = event.cpuUsage;
            features[1] = event.memoryUsage;
            features[2] = event.processCount;
            
            // Derived features
            features[3] = event.cpuUsage * event.memoryUsage; // interaction
            features[4] = event.cpuUsage / (event.processCount + 1); // normalization
            features[5] = event.memoryUsage / (event.processCount + 1);
            
            // Action encoding
            int actionCode = hashAction(event.action);
            features[6] = actionCode;
            
            // User type encoding
            int userCode = hashUser(event.userName);
            features[7] = userCode;
            
            // Fill remaining features with placeholder values
            // In production, would extract more sophisticated features
            for (int i = 8; i < 73; i++) {
                features[i] = Math.random() * 100; // placeholder
            }
            
            return new FeatureVector(event.timestamp, event.nodeName, features);
        }
        
        private int hashAction(String action) {
            return Math.abs(action.hashCode()) % 20;
        }
        
        private int hashUser(String user) {
            return Math.abs(user.hashCode()) % 50;
        }
    }
    
    /**
     * Aggregation function for windowed statistics
     */
    public static class WindowStatisticsAggregator {
        public static double[] aggregateFeatures(List<FeatureVector> vectors) {
            if (vectors.isEmpty()) {
                return new double[73];
            }
            
            double[] aggregated = new double[73];
            
            // Calculate mean for each feature
            for (int i = 0; i < 73; i++) {
                double sum = 0;
                for (FeatureVector v : vectors) {
                    sum += v.features[i];
                }
                aggregated[i] = sum / vectors.size();
            }
            
            // Add variance and other statistics
            for (int i = 0; i < Math.min(73, 73/2); i++) {
                double mean = aggregated[i];
                double variance = 0;
                for (FeatureVector v : vectors) {
                    variance += Math.pow(v.features[i] - mean, 2);
                }
                aggregated[i + 37] = variance / vectors.size();
            }
            
            return aggregated;
        }
    }
    
    public static void main(String[] args) throws Exception {
        System.out.println("=" + "=".repeat(70));
        System.out.println("KubeShield - Apache Flink Streaming Feature Engineering");
        System.out.println("=" + "=".repeat(70));
        
        // Set up Flink streaming environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(4); // Parallelism for distributed processing
        
        System.out.println("\n[CONFIG] Initializing Kafka consumer...");
        System.out.println("  - Input Topic: kubeshield-telemetry");
        System.out.println("  - Bootstrap Servers: localhost:9092");
        System.out.println("  - Consumer Group: kubeshield-feature-extractor");
        
        // Kafka consumer properties
        Properties kafkaProps = new Properties();
        kafkaProps.setProperty("bootstrap.servers", "localhost:9092");
        kafkaProps.setProperty("group.id", "kubeshield-feature-extractor");
        kafkaProps.setProperty("auto.offset.reset", "earliest");
        
        // Create Kafka source
        FlinkKafkaConsumer<String> kafkaConsumer = new FlinkKafkaConsumer<>(
            "kubeshield-telemetry",
            new SimpleStringSchema(),
            kafkaProps
        );
        
        DataStream<String> kafkaStream = env.addSource(kafkaConsumer);
        
        System.out.println("\n[PIPELINE] Setting up feature extraction pipeline...");
        
        // Feature extraction
        DataStream<FeatureVector> featureStream = kafkaStream
            .map(new FeatureExtractionMapper())
            .name("Extract Features");
        
        System.out.println("  ✓ Feature extraction mapper configured (73 features)");
        
        // Windowed aggregation (1-minute tumbling window)
        DataStream<String> windowedFeatures = featureStream
            .keyBy(f -> f.nodeName)
            .timeWindow(Time.minutes(1))
            .apply((String key, TimeWindow window, Iterable<FeatureVector> values, org.apache.flink.streaming.api.functions.windowing.WindowFunction<FeatureVector, String, String, TimeWindow> out) -> {
                List<FeatureVector> vectorList = new ArrayList<>();
                values.forEach(vectorList::add);
                
                double[] aggregated = WindowStatisticsAggregator.aggregateFeatures(vectorList);
                
                JSONObject result = new JSONObject();
                result.put("timestamp", window.getEnd());
                result.put("node_name", key);
                result.put("window_size", vectorList.size());
                result.put("features", aggregated);
                
                out.collect(result.toString());
            })
            .name("Aggregate Features");
        
        System.out.println("  ✓ Time window aggregation configured (60s tumbling)");
        System.out.println("  ✓ Node-level key partitioning enabled");
        
        // Kafka sink to send processed features
        FlinkKafkaProducer<String> kafkaProducer = new FlinkKafkaProducer<>(
            "kubeshield-features",
            new SimpleStringSchema(),
            kafkaProps
        );
        
        windowedFeatures.addSink(kafkaProducer).name("Send to Kafka");
        
        System.out.println("\n[OUTPUT] Kafka producer configured");
        System.out.println("  - Output Topic: kubeshield-features");
        System.out.println("  - Format: JSON feature vectors");
        
        System.out.println("\n[EXECUTION] Starting Flink job...");
        System.out.println("  Consuming from: kubeshield-telemetry");
        System.out.println("  Publishing to: kubeshield-features");
        System.out.println("  Processing: Continuous streaming (60s windows)");
        System.out.println("\n" + "=".repeat(70));
        
        // Execute the pipeline
        env.execute("KubeShield Feature Engineering");
    }
}
