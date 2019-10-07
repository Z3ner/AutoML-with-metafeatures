import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

def featuresMean(features):
	return [np.mean(feature) for feature in features]

def featuresStd(features):
	return [np.std(feature) for feature in features]

def featuresSkew(features):
	return [skew(feature) for feature in features]

def featuresKurt(features):
	return [kurtosis(feature) for feature in features]

def nFeatures(features):
	return len(features)

def metafeatures(stats):
	mf_mean = np.mean(stats)
	mf_std = np.std(stats)
	mf_min, mf_q1, mf_q2, mf_q3, mf_max = np.percentile(stats, [0,25,50,75,100])
	mf_skew = skew(stats)
	mf_kurtosis = kurtosis(stats)
	return mf_mean, mf_std, mf_min, mf_q1, mf_q2, mf_q3, mf_max, mf_skew, mf_kurtosis


def metadataset_row(x_data):

	feature_transformations = [featuresMean, featuresStd, featuresSkew, featuresKurt]
	features = np.transpose(x_data)

	metafeatures_row = [metafeatures(ft(features)) for ft in feature_transformations]
	#metafeatures_row += [nFeatures(features)]
	metafeatures_row = np.asarray(metafeatures_row)
	return metafeatures_row.flatten()