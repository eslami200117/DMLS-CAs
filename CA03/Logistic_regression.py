from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, min, mean, stddev, variance
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml import Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql.functions import udf
from pyspark.ml.feature import Imputer

if __name__ == "__main__":
  spark = SparkSession\
    .builder\
    .appName("LogisticRegression")\
    .getOrCreate()
  sc = spark.sparkContext

csv_file_path = "heart.csv"
df = spark.read.csv(f"hdfs://172.18.35.204:9000/{csv_file_path}", header=True)
# df = spark.read.csv(csv_file_path, header=True)
df = df.select([col(c).cast("float") for c in df.columns])
train_df, test_df = df.randomSplit(weights=[0.85,0.15], seed=100)

vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
cholScaler = StandardScaler(inputCol="chol", outputCol="chol")
trtbpsScaler = StandardScaler(inputCol="trtbps", outputCol="trtbps")
thalachhScaler = StandardScaler(inputCol="thalachh", outputCol="thalachh")
ageScaler = StandardScaler(inputCol="age", outputCol="age")

target_column = train_df.columns[-1]



lr = LogisticRegression(maxIter=10, regParam=0.001, labelCol=target_column)
vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
train_df = train_df.withColumn("age_", vector_udf(df["age"]))

ageScaler = StandardScaler(inputCol="age_", outputCol="age")
pipeline = Pipeline(stages=[ageScaler, lr])
model = pipeline.fit(train_df)

  

