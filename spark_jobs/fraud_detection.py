from pyspark.sql import SparkSession

def run_spark_job(input_path: str, output_path: str):
    """
    Run Spark fraud detection and save results.
    """
    spark = SparkSession.builder \
        .appName("FraudDetectX Spark Job") \
        .getOrCreate()

    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Simple fraud detection rules
    from pyspark.sql.functions import col, when

    df = df.withColumn(
        "FraudFlag",
        when(col("AadhaarFlag") | col("ClaimFlag"), True).otherwise(False)
    )

    df.write.mode("overwrite").option("header", True).csv(output_path)
    print(f"✅ Spark fraud detection completed → {output_path}")

    spark.stop()
