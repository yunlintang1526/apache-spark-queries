import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR
from pyspark.ml.stat import Summarizer


# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics 
# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # step 1&2
    # use agg to calculate the mean and count
    gr = review_data.groupBy(review_data.asin)
    agg_df = gr.agg({'overall':'mean', 'asin':'count'})
    # rename the table
    agg_df = agg_df.withColumnRenamed('avg(overall)', 'meanRating').withColumnRenamed('count(asin)', 'countRating')
    # join the agg_df to product_data
    p_asin = product_data.select(product_data['asin'])
    product_data = p_asin.join(agg_df, on='asin', how='left')
    
    # step 3
    count_row = product_data.count()
    
    mean_mr = product_data.select(F.avg(F.col('meanRating'))).head()[0]
    var_mr = product_data.select(F.var_samp(F.col('meanRating'))).head()[0]
    numNulls_mr = product_data.filter(product_data['meanRating'].isNull()).count()
    
    mean_cr = product_data.select(F.avg(F.col('countRating'))).head()[0]
    var_cr = product_data.select(F.var_samp(F.col('countRating'))).head()[0]
    numNulls_cr = product_data.filter(product_data['countRating'].isNull()).count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:

    res['count_total'] = count_row
    res['mean_meanRating'] = mean_mr
    res['variance_meanRating'] = var_mr
    res['numNulls_meanRating'] = numNulls_mr
    res['mean_countRating'] = mean_cr
    res['variance_countRating'] = var_cr
    res['numNulls_countRating'] = numNulls_cr


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    # TODO: can we use alias?
    p = product_data.alias('p')

    # step 1
    p = p.withColumn('category', F.flatten(p.categories))
    p = p.withColumn('category', p['category'].getItem(0))
    p = p.replace({'': None}, subset=['category'])
    
    # step 2
    p = p.withColumn('bestSalesCategory', F.map_keys(F.col('salesRank')).getItem(0))
    p = p.withColumn('bestSalesRank', F.map_values(F.col('salesRank')).getItem(0))
    
    product_data = p
    
    # step 3
    cnt_total = product_data.count()
    mean_bsr = product_data.select(F.avg(F.col('bestSalesRank'))).head()[0]
    var_bsr = product_data.select(F.var_samp(F.col('bestSalesRank'))).head()[0]
    numNulls_cat = product_data.filter(product_data['category'].isNull()).count()
    cntDis_cat = product_data.select(F.countDistinct(product_data.category)).head()[0]
    numNulls_bsc = product_data.filter(product_data['bestSalesCategory'].isNull()).count()
    cntDis_bsc = product_data.select(F.countDistinct(product_data['bestSalesCategory'])).head()[0]



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
    
    res['count_total'] = cnt_total
    res['mean_bestSalesRank'] = mean_bsr
    res['variance_bestSalesRank'] = var_bsr
    res['numNulls_category'] = numNulls_cat
    res['countDistinct_category'] = cntDis_cat
    res['numNulls_bestSalesCategory'] = numNulls_bsc
    res['countDistinct_bestSalesCategory'] = cntDis_bsc



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    price = product_data.select('asin', 'price')
    price = price.withColumnRenamed('asin', 'priceAsin')
    p = product_data.alias('p')
    p = p.select('asin', 'related')
    

    # step 1
    p = p.withColumn('also_viewed', p['related']['also_viewed'])
    expl = p.select('asin', F.explode_outer(p['also_viewed']))
    expl = expl.withColumnRenamed('col', 'avAsin')
    joined = expl.join(price, expl['avAsin'] == price['priceAsin'], how='left')
    
    gr = joined.groupBy(joined.asin)
    agg_df = gr.agg({'price':'mean'})
    p1 = agg_df.withColumnRenamed('avg(price)', 'meanPriceAlsoViewed')
    
    # step 2: add length column
    p2 = p.withColumn('countAlsoViewed', F.size(p['also_viewed']))
    p2 = p2.replace({-1: None}, subset=['countAlsoViewed'])

    # step 3
    cnt_total = p1.count()
    mean_mpav = p1.select(F.avg(F.col('meanPriceAlsoViewed'))).head()[0]
    var_mpav = p1.select(F.var_samp(F.col('meanPriceAlsoViewed'))).head()[0]
    numNulls_mpav = p1.filter(p1['meanPriceAlsoViewed'].isNull()).count()
    mean_cav = p2.select(F.avg(F.col('countAlsoViewed'))).head()[0]
    var_cav = p2.select(F.var_samp(F.col('countAlsoViewed'))).head()[0]
    numNulls_cav = p2.filter(p2['countAlsoViewed'].isNull()).count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    
    res["count_total"] = cnt_total
    
    res["mean_meanPriceAlsoViewed"] =  mean_mpav
    res['variance_meanPriceAlsoViewed'] =  var_mpav
    res['numNulls_meanPriceAlsoViewed'] =  numNulls_mpav
    res['mean_countAlsoViewed'] = mean_cav
    res['variance_countAlsoViewed'] = var_cav
    res['numNulls_countAlsoViewed'] =  numNulls_cav



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    p = product_data.alias('p')
    
    # Step 1
    p = p.withColumn('price', p['price'].cast('float'))
    mean_price = p.select(F.avg(F.col('price'))).head()[0]
    p = p.withColumn('meanImputedPrice', p['price'])

    # Step 2
    median_price = p.approxQuantile('price', [0.5], 0.001)[0]
    p = p.withColumn('medianImputedPrice', p['price'])
    
    # Step 3
    p = p.withColumn('unknownImputedTitle', p.title)
    p = p.replace({'':'unknown'}, subset=['unknownImputedTitle'])

    # fill nulls
    p = p.fillna({'medianImputedPrice': median_price, 
                  'meanImputedPrice': mean_price,
                  'unknownImputedTitle': 'unknown'})
    
    product_data = p
    
    # Step 4
    cnt_total = product_data.count()
    mean_mip = product_data.select(F.avg(F.col('meanImputedPrice'))).head()[0]
    var_mip = product_data.select(F.var_samp(F.col('meanImputedPrice'))).head()[0]
    numNulls_mip = product_data.filter(product_data['meanImputedPrice'].isNull()).count()
    
    mean_meip = product_data.select(F.avg(F.col('medianImputedPrice'))).head()[0]
    var_meip = product_data.select(F.var_samp(F.col('medianImputedPrice'))).head()[0]
    numNulls_meip = product_data.filter(product_data['medianImputedPrice'].isNull()).count()
    numUnk = product_data.filter(product_data['unknownImputedTitle'] == 'unknown').count()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    
    res['count_total'] = cnt_total
    res['mean_meanImputedPrice'] = mean_mip
    res['variance_meanImputedPrice'] = var_mip
    res['numNulls_meanImputedPrice'] = numNulls_mip
    res['mean_medianImputedPrice'] = mean_meip
    res['variance_medianImputedPrice'] = var_meip
    res['numNulls_medianImputedPrice'] = numNulls_meip
    res['numUnknowns_unknownImputedTitle'] = numUnk
    



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    p = product_processed_data.alias('p')
    
    # step 1
    p = p.withColumn('titleArray', F.lower(p['title']))
    p = p.withColumn('titleArray', F.split(p['titleArray'], '\s'))
    
    # step 2
    w2v = M.feature.Word2Vec(minCount=100, vectorSize=16, seed=SEED, 
                             numPartitions=4, inputCol='titleArray', outputCol='vec')
    model = w2v.fit(p)
    
    
    product_processed_data_output = p



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    
    p = product_processed_data.alias('p')
    
    # step 1
    # convert category column to numerical indices
    indx = M.feature.StringIndexer(inputCol = 'category', outputCol='indexed')
    indx_model = indx.fit(p)
    p = indx_model.transform(p)
    # one-hot encoding
    ohe = M.feature.OneHotEncoderEstimator(inputCols=['indexed'], 
                                          outputCols=['categoryOneHot'], dropLast=False)
    ohe_model = ohe.fit(p)
    p = ohe_model.transform(p)
    
    # step 2
    pca = M.feature.PCA(inputCol='categoryOneHot', outputCol='categoryPCA', k=15)
    pca_model = pca.fit(p)
    p = pca_model.transform(p)
    
    # step 4
    cnt_total = p.count()
    summ = Summarizer.metrics('mean')
    mean_onehot = p.select(summ.summary(p.categoryOneHot)).head()[0][0]
    mean_onehot = mean_onehot.values.tolist()
    mean_pca = p.select(summ.summary(p.categoryPCA)).head()[0][0]
    mean_pca = mean_pca.values.tolist()




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:

    res['count_total'] = cnt_total
    res['meanVector_categoryOneHot'] = mean_onehot
    res['meanVector_categoryPCA'] = mean_pca


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    # step 1
    dt = M.regression.DecisionTreeRegressor(labelCol='overall', maxDepth=5)
    dt_model = dt.fit(train_data)
    
    # step 2
    test_data_pred = dt_model.transform(test_data)
    evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction',
                                                labelCol='overall', metricName='rmse')
    rmse = evaluator.evaluate(test_data_pred)
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:

    res['test_rmse'] = rmse

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------

    train, test = train_data.randomSplit([0.75, 0.25], seed=12345)
    para_array = [5,7,9,12]
    rmse_array = []
    model_array = []
    for x in para_array:
        dt = M.regression.DecisionTreeRegressor(labelCol='overall', maxDepth=x)
        dt_model = dt.fit(train)
        test_data_pred = dt_model.transform(test)
        evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction',
                                                labelCol='overall', metricName='rmse')
        rmse = evaluator.evaluate(test_data_pred)
        rmse_array.append(rmse)
        model_array.append(dt_model)
    
    # use the best
    index = rmse_array.index(min(rmse_array))
    best_model = model_array[index]
    
    test_data_pred = best_model.transform(test_data)
    evaluator = M.evaluation.RegressionEvaluator(predictionCol='prediction',
                                                labelCol='overall', metricName='rmse')
    rmse_best = evaluator.evaluate(test_data_pred)
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:

    res["test_rmse"] =  rmse_best
    res["valid_rmse_depth_5"] = rmse_array[0]
    res["valid_rmse_depth_7"] = rmse_array[1]
    res["valid_rmse_depth_9"] = rmse_array[2]
    res["valid_rmse_depth_12"] = rmse_array[3]
    
    

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

