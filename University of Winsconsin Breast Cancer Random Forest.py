# Databricks notebook source
# MAGIC %md
# MAGIC # University of Wisconsin Breast Cancer Random Forest
# MAGIC This analysis seeks to use PySpark to analyze a data set of features of breast masses.  The features were documented based on images from the fine needle aspiration of the mass, and desribe the cell nucleus.  The outcome label, located in the second column, is B - Benign, or M - malignant.  The data was donated by the University of Wisconsin and is located on the UCI Machine Learning Repository (https://archive.ics.uci.edu/datasets) and listed under "Breast Cancer Wisconsin (Diagnostic)."
# MAGIC
# MAGIC Below are the fields in the data set:
# MAGIC 1. ID number
# MAGIC 2. Diagnosis (M = malignant, B = benign)
# MAGIC
# MAGIC 3-32. Ten real-valued features are computed for each cell nucleus (each one list listed 3 times):
# MAGIC
# MAGIC 	a) radius (mean of distances from center to points on the perimeter)
# MAGIC 	b) texture (standard deviation of gray-scale values)
# MAGIC 	c) perimeter
# MAGIC 	d) area
# MAGIC 	e) smoothness (local variation in radius lengths)
# MAGIC 	f) compactness (perimeter^2 / area - 1.0)
# MAGIC 	g) concavity (severity of concave portions of the contour)
# MAGIC 	h) concave points (number of concave portions of the contour)
# MAGIC 	i) symmetry 
# MAGIC 	j) fractal dimension ("coastline approximation" - 1)
# MAGIC
# MAGIC
# MAGIC Cancer touches so many, and robs many families of years together.  As data scientists and researchers, helping to refine early identification and removal of cancer can be lifesaving for people.  Further, accurate identification of benign and malignant tumors can be a life or death situation, and the ability to provide predictive analytics/confirmational support for human pathologists could be beneficial.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and Examine Data
# MAGIC The first step for our analysis is to load in the data set.  Before we do this, we need to indicate the schema for our file so that we have each feature properly labeled.  As noted above, our data has many dimensions and features and a schema ensures we have data accurately loaded and can best identify the most important features.
# MAGIC

# COMMAND ----------

sc = spark.sparkContext
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType
schema = schema = StructType( [\
                        StructField('ID', StringType(), True), \
                        StructField('diagnosis', StringType(), True), \
                        StructField('radius1', DoubleType(), True), \
                        StructField('texture1', DoubleType(), False), \
                        StructField('perimeter1', DoubleType(), False), \
                        StructField('area1', DoubleType(), False), \
                        StructField('smoothness1', DoubleType(), False), \
                        StructField('compactness1', DoubleType(), False), \
                        StructField('concavity1', DoubleType(), False), \
                        StructField('concave_points1', DoubleType(), False), \
                        StructField('symmetry1', DoubleType(), False), \
                        StructField('fractal_dimension1', DoubleType(), False), \
                        StructField('radius2', DoubleType(), True), \
                        StructField('texture2', DoubleType(), False), \
                        StructField('perimeter2', DoubleType(), False), \
                        StructField('area2', DoubleType(), False), \
                        StructField('smoothness2', DoubleType(), False), \
                        StructField('compactness2', DoubleType(), False), \
                        StructField('concavity2', DoubleType(), False), \
                        StructField('concave_points2', DoubleType(), False), \
                        StructField('symmetry2', DoubleType(), False), \
                        StructField('fractal_dimension2', DoubleType(), False), \
                        StructField('radius3', DoubleType(), True), \
                        StructField('texture3', DoubleType(), False), \
                        StructField('perimeter3', DoubleType(), False), \
                        StructField('area3', DoubleType(), False), \
                        StructField('smoothness3', DoubleType(), False), \
                        StructField('compactness3', DoubleType(), False), \
                        StructField('concavity3', DoubleType(), False), \
                        StructField('concave_points3', DoubleType(), False), \
                        StructField('symmetry3', DoubleType(), False), \
                        StructField('fractal_dimension3', DoubleType(), False) \
                           ])

df = spark.read.format("csv").\
    option("header", False).\
    schema(schema).\
    load("/FileStore/tables/wdbc.csv")

df.show(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create training and test set
# MAGIC Now that we have loaded our full data set with field names into a Spark dataframe, we can split the data into training and test data.  We are using 70% of the data for training and 30% of it for testing.  We set a random seed in order to ensure the results are reproducible.
# MAGIC

# COMMAND ----------

test, train = df.randomSplit([0.3, 0.7], seed = 406)
train.count()
# test.count()

# We have 392 rows in our training data set and 177 in our test data set

# COMMAND ----------

train.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Pipeline and Fit to Data
# MAGIC
# MAGIC The next thing we need to do is create a pipeline to prepare our data for fitting our model, and then finally pass it to our model.  A pipeline allows sequential modification of the data in order to ensure that the same steps are completed each time on each row of data.  Creating a pipeline allows us to also ensure that the training and test sets are treated the same.  The first step in our pipeline is a StringIndexer, which takes our diagnosis column (containing M for malignant or B for benign and converts it to a numerical representation that our model can use, in this case, it assigns indexes 0 and 1 depending on the value in a given row) and places this value into a new column titled "label."  The next step in our pipeline is a VectorAssembler.  This function takes the input features of each row and creates a vector version of them, and places that value in a new column titled "features."  Both of these steps are necessary to prepare the data for our model. 
# MAGIC After data preparation, our pipeline ends in our model, and at this point our data is fit to the model.  In order for our model to fit the data, we must create the model variable and indicate what columns will be used for the label, features, and number of trees.  The model I have chosen is a Random Forest Classifier.  I chose this model as it is a good model for high dimensional data.  A random forest creates many decision trees and the importance of the feature from each tree is averaged to provide the final feature importance.  The parameter, numTrees, tells the model how many trees to assemble.  A decision trees starts with a single data feature and uses it to split the data based on whether evaluation of the feature is True or False, and iterates many times through the features. much like a flow chart.  An example with our data for a first node split would be whether or not radius1 is greater than or less than a certain value.  At the next node, it would evaluate another feature, and on and on.  As noted, a random forest is an aggregation of many decision trees, and the average importance of a feature is used in the final evaluation of data. 
# MAGIC
# MAGIC Below we create our indexer for our label (outcome) column, create our vector of features for input into our model, instantiate our model, and finally pass out training data to our pipeline, fitting it to our model.
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Binarize label column
indexer = StringIndexer(inputCol = "diagnosis", outputCol = "label")

# Create a vector assembler of features
# We need to create a vector of all our features for each sample so that we can create a single column "features," which we will feed into our model
input_features = ['radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', \
    'symmetry1', 'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2', 'compactness2', 'concavity2', \
    'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', \
    'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']
v_assembler = VectorAssembler(inputCols = input_features, outputCol = "features")

# Create a random forest model
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

# Create pipeline, which our data will pass through step by step
pipeline = Pipeline(stages = [indexer, v_assembler, rf])
model = pipeline.fit(train)



# COMMAND ----------

static_prediction = model.transform(test)
# prediction.show()
selected = static_prediction.select("id", "rawPrediction", "probability", "prediction", "label")
for row in selected.collect():
    print(row)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Test Data Analysis
# MAGIC
# MAGIC With our random forest classifier fit to our training data, we can now address our test data.  In order to view the training data as historical data which can be used as a predictor of new data as it arrives, we need to incorporate streaming into the analysis (think of all the biopsies occurring at each oncology practice or hospital throughout the world being documented and fed into the model in real time.)
# MAGIC
# MAGIC We can simulate streaming data for our test set by separating the data into many partitions (subsets) of data at a time.
# MAGIC

# COMMAND ----------

# We already have a testing dataframe, which we can convert to smaller files to simulate streaming data.

test_df = test.repartition(25).persist()

# Write out files, one per partition to a new folder
dbutils.fs.rm("FileStore/tables/wdbc/", True) # remove contents of folder
test_df.write.format("csv").option("header", True).save("dbfs:/FileStore/tables/wdbc/")

# COMMAND ----------

# Create source and sink for streaming data
source_stream = spark.readStream.format("csv").option("header", True).schema(schema).option("maxFilesPerTrigger", 1).load("dbfs:/FileStore/tables/wdbc/")
# Transform test set with pipeline and model
prediction = model.transform(source_stream)
selected = prediction.select("id", "rawPrediction", "probability", "prediction", "label")
sink_stream = selected.writeStream.outputMode("append").format("memory").queryName("breast_cancer").start()

import time
time.sleep(5)
current = spark.sql("SELECT * from breast_cancer")
current.show(truncate = False)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Model Evaluation and Conclusion
# MAGIC
# MAGIC We fit our model to our test data, both in a static single fit of all of the test data, and in a streaming query, where the continuous input of data was used to make predictions on each row as it arrived.  To evaluate the overall Random Forest Classifier performance on the test data set, I am using a binary classification evaluator.  It takes the known label column (Malignant or Benign from the original dataset) and the prediction column from our fitted test data output and compares them.  In classification models such as ours, the Receiver Operating Characteristic, which compares true positives and false negatives using the known values and predictions.  Next, it computes the area under the curve of these values, which can range from 0 to 1.  1 is an ideal value.
# MAGIC

# COMMAND ----------


from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = 'areaUnderROC')
auc = evaluator.evaluate(static_prediction)
print(f'The area under the ROC is: {auc}')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Based on our ROC AUC value of approximately 0.95, our dataset and Random Forest Classifier model were a good predictor for our test data. The model could be further optimized to improve the ROC AUC even more, however our focus was on creating a streaming model that could provide "real-time" feedback to surgeons and pathologists.  The main challenge with this dataset was the number of features and attempting to review output given the way they displayed.  Further optimization of the method could be done via dimensionality reduction, which could help with this, and the evaluation of the model could be done in a streaming step as well, however depending on the use of the model (machine learning prediction prior to pathologist's final report or machine learning confirmation after final report), labels may not be known. 
# MAGIC
