import numpy as np
import matplotlib.pyplot as plt
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats, optimize

# Create plot with given data and labels
# Used for plots of data (Bitcoin prices or sentiments) over time
def create_plot(xData1, yData1, color1, legend1, xLabel, yLabel, title, fname, yData2=None, color2=None, legend2=None):
  plt.plot(xData1, yData1, color = color1, label=legend1)
  if yData2 and color2 and legend2:
    plt.plot(xData1, yData2, color=color2, label=legend2)
  
  plt.xticks(rotation = 45, fontsize = 4)
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.title(title)
  plt.legend()
  plt.savefig(fname)
  plt.show()

# Function for exponential fit
def func(x, a, b, c):
  return a * x ** b + c

# Create scatter plot with given data and labels
# Used for correlation plots between sentiment and Bitcoin prices
def create_scatter_plot(xData, yData, xLabel, yLabel, title, fname):
  # Create linear regression
  slope, intercept, r_value, p_value, std_err = stats.linregress(xData, yData)
  regression = np.linspace(-1,1) * slope + intercept

  yMax = max(regression)
  plt.annotate(f'p-value: {p_value}', (0, yMax))
  plt.annotate(f'R^2: {r_value * r_value}', (0, yMax * .93))
  plt.annotate(f'slope: {slope}', (0, yMax * .86))
  plt.annotate(f'standard error: {std_err}', (0, yMax * .79))

  # Create Spearman rank coefficient
  cor, pval = stats.spearmanr(xData, yData)
  plt.annotate(f'Spearman rank coefficient: {cor}', (0, yMax * .72))

  # Create non-linear least squares fit
  popt, pcov = optimize.curve_fit(func, xData, yData)
  plt.plot(xData, func(xData, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

  plt.xticks(rotation = 45, fontsize = 4)
  plt.scatter(xData, yData)
  plt.plot(np.linspace(-1,1), regression)
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.title(title)
  plt.savefig(fname)
  plt.show()

# Return sentiment score of a given Reddit comment
def get_sentiment(analyzer, comment):
  return analyzer.polarity_scores(comment)['compound']

# Return mean score of Reddit comments in a given time interval
def get_mean(analyzer, interval):
  return sum([get_sentiment(analyzer, comment) for comment in interval]) / len(interval)

# Return median score of Reddit comments in a given time interval
def get_median(analyzer, interval):
  interval.sort()
  mid = len(interval) // 2
  return (interval[mid] + interval[mid - 1]) / 2 if len(interval) % 2 == 0 else interval[mid]

  return sum([get_sentiment(analyzer, comment) for comment in interval])/len(interval)

# Turn json comments to json sentiments
def comments_to_sentiments(analyzer, fname, output_fname):
  comments = get_json(fname)
  output = {}

  for time in comments:
    output[time] = get_mean(analyzer, comments[time])
  
  with open(output_fname, 'w') as fp:
    json.dump(output, fp)

# Return json file in dictionary format
def get_json(fname):
  with open(fname) as json_file:
    return json.load(json_file)

# Format json date keys to YYYY-MM-DD
def format_json(fname, output_fname):
  data = get_json(fname)
  output = {}

  for time in data:
    year, month, day = time.split("-")
    if len(month) == 1:
      month = "0" + month
    if len(day) == 1:
      day = "0" + day

    output[year + "-" + month + "-" + day] = data[time]
  
  with open(output_fname, 'w') as fp:
    json.dump(output, fp)


# Normalize data
def normalize(arr):
  minimum = min(arr)
  maximum = max(arr)
  return [(x - minimum) / (maximum - minimum) for x in arr]

if __name__ == "__main__":
  analyzer = SentimentIntensityAnalyzer()

  # # Sample plots
  # x = [1, 2, 3, 4]
  # y = [
  #   get_sentiment(analyzer, "Wow this movie was mediocre"),
  #   get_sentiment(analyzer, "Congratulations, great job!"),
  #   get_sentiment(analyzer, "Omg I hate you so much ughhhhh"),
  #   get_sentiment(analyzer, "This is the worst day of my life")
  # ]

  # y2 = [.2, .2, .2, .2]
  # create_plot(x, y, "darkorange", "Sentiment", "X Label", "Y Label", "Plot Title", "sampleplot.png", y2, "lightblue", "Random Stuff")

  # x2 = [0.5,-0.2,-0.8,0.2]
  # create_scatter_plot(x2, y, "X Label", "Y Label", "Plot Title", "sampleplot2.png")

  ##########################################
  # First run of analyses of month/year data
  ##########################################

  # # Extract data from json files
  # btc_month = get_json('month.json')
  # btc_year = get_json('year.json')

  # # Run once to extract average sentiments from comments
  # # comments_to_sentiments(analyzer, 'by_month.json', 'sentiment_month.json')
  # # comments_to_sentiments(analyzer, 'by_year.json', 'sentiment_year.json')

  # sentiment_month = get_json('sentiment_month.json')
  # sentiment_year = get_json('sentiment_year.json')

  # # Get keys in sorted order for plotting
  # months = sorted(btc_month.keys())
  # years = sorted(btc_year.keys())

  # # Ensure months and years correspond between bitcoin prices and sentiment data
  # assert months == sorted(sentiment_month.keys())
  # assert years == sorted(sentiment_year.keys())

  # btc_month_sorted = [btc_month[key] for key in months]
  # btc_year_sorted = [btc_year[key] for key in years]
  # sentiment_month_sorted = [sentiment_month[key] for key in months]
  # sentiment_year_sorted = [sentiment_year[key] for key in years]

  # btc_month_sorted_normalized = normalize(btc_month_sorted)
  # btc_year_sorted_normalized = normalize(btc_year_sorted)
  # sentiment_month_sorted_normalized = normalize(sentiment_month_sorted)
  # sentiment_year_sorted_normalized = normalize(sentiment_year_sorted)

  # # # Plot of Bitcoin prices over time (months)
  # create_plot(months, btc_month_sorted, "darkorange", "Bitcoin Price", "Month", "Price ($)", "Bitcoin Prices Over Months", "btc_month.png")

  # # # Plot of Bitcoin prices over time (years)
  # create_plot(years, btc_year_sorted, "darkorange", "Bitcoin Price", "Year", "Price ($)", "Bitcoin Prices Over Years", "btc_year.png")

  # # # Plot of Reddit comment sentiments over time (months)
  # create_plot(months, sentiment_month_sorted, "darkorange", "Sentiment", "Month", "Sentiment (0 - 1)", "Reddit Comment Sentiment Over Months", "sentiment_month.png")

  # # # Plot of Reddit comment sentiments over time (years)
  # create_plot(years, sentiment_year_sorted, "darkorange", "Sentiment", "Year", "Sentiment (0 - 1)", "Reddit Comment Sentiment Over Years", "sentiment_year.png")

  # # Plot of Bitcoin price vs. sentiment (month)
  # create_scatter_plot(sentiment_month_sorted, btc_month_sorted, "Sentiment", "Bitcoin Price", "Bitcoin Prices vs. Sentiment (month)", "regression_month.png")

  # # Plot of Bitcoin price vs. sentiment (year)
  # create_scatter_plot(sentiment_year_sorted, btc_year_sorted, "Sentiment", "Bitcoin Price", "Bitcoin Prices vs. Sentiment (year)", "regression_year.png")

  
  # # Normalized plots
  # # Plot of normalized Bitcoin price and sentiment versus month
  # create_plot(months, btc_month_sorted_normalized, "darkorange", "Bitcoin Price", "Month", "Normalized Price/Sentiment", "Normalized Price/Sentiment vs. Month", "normalized_month.png", sentiment_month_sorted_normalized, "lightblue", "Sentiment")
  
  # # Plot of normalized Bitcoin price and sentiment versus year
  # create_plot(years, btc_year_sorted_normalized, "darkorange", "Bitcoin Price", "Year", "Normalized Price/Sentiment", "Normalized Price/Sentiment vs. Year", "normalized_year.png", sentiment_year_sorted_normalized, "lightblue", "Sentiment")

  # # Plot of Bitcoin price vs. sentiment normalized (month)
  # create_scatter_plot(sentiment_month_sorted_normalized, btc_month_sorted_normalized, "Sentiment", "Bitcoin Price", "Normalized Bitcoin Prices vs. Sentiment (month)", "normalized_regression_month.png")

  # # Plot of Bitcoin price vs. sentiment normalized (year)
  # create_scatter_plot(sentiment_year_sorted_normalized, btc_year_sorted_normalized, "Sentiment", "Bitcoin Price", "Normalized Bitcoin Prices vs. Sentiment (year)", "normalized_regression_year.png")

  ###################################
  # Next run of analyses of days data
  ###################################

    # Extract data from json files
  btc_day = get_json('days.json')

  # Run once to extract average sentiments from comments
  # comments_to_sentiments(analyzer, 'by_day.json', 'sentiment_day.json')
  # comments_to_sentiments(analyzer, 'by_day_bitcoin.json', 'sentiment_day_bitcoin.json')

  sentiment_day = get_json('sentiment_day.json')
  sentiment_day_bitcoin = get_json('sentiment_day_bitcoin.json')

  # Get keys in sorted order for plotting
  days = sorted(btc_day.keys())

  # Ensure days correspond between bitcoin prices and sentiment data
  assert days == sorted(sentiment_day.keys())
  assert days == sorted(sentiment_day_bitcoin.keys())

  btc_day_sorted = [btc_day[key] for key in days]
  sentiment_day_sorted = [sentiment_day[key] for key in days]
  sentiment_day_bitcoin_sorted = [sentiment_day_bitcoin[key] for key in days]

  btc_day_sorted_normalized = normalize(btc_day_sorted)
  sentiment_day_sorted_normalized = normalize(sentiment_day_sorted)
  sentiment_day_bitcoin_sorted_normalized = normalize(sentiment_day_bitcoin_sorted)

  # # Plot of Bitcoin prices over time (days)
  create_plot(days, btc_day_sorted, "darkorange", "Bitcoin Price", "Day", "Price ($)", "Bitcoin Prices Over Days", "btc_day.png")

  # # Plot of Reddit comment sentiments over time (days)
  create_plot(days, sentiment_day_sorted, "darkorange", "Sentiment", "Day", "Sentiment (0 - 1)", "Reddit Comment Sentiment Over Days", "sentiment_day.png")

  # # Plot of Reddit comment sentiments over time (days and bitcoin subreddit only)
  create_plot(days, sentiment_day_bitcoin_sorted, "darkorange", "Sentiment", "Day", "Sentiment (0 - 1)", "Reddit Comment Sentiment Over Days - Bitcoin Subreddit", "sentiment_day_bitcoin.png")

  # Plot of Bitcoin price vs. sentiment (day)
  create_scatter_plot(sentiment_day_sorted, btc_day_sorted, "Sentiment", "Bitcoin Price", "Bitcoin Prices vs. Sentiment (day)", "regression_day.png")

  # Plot of Bitcoin price vs. sentiment (day and bitcion subreddit only)
  create_scatter_plot(sentiment_day_bitcoin_sorted, btc_day_sorted, "Sentiment", "Bitcoin Price", "Bitcoin Prices vs. Sentiment (day) - Bitcoin Subreddit", "regression_day_bitcoin.png")

  
  # Normalized plots
  # Plot of normalized Bitcoin price and sentiment versus day
  create_plot(days, btc_day_sorted_normalized, "darkorange", "Bitcoin Price", "Day", "Normalized Price/Sentiment", "Normalized Price/Sentiment vs. Day", "normalized_day.png", sentiment_day_sorted_normalized, "lightblue", "Sentiment")
  
  # Plot of normalized Bitcoin price and sentiment versus day - bitcoin subreddit only
  create_plot(days, btc_day_sorted_normalized, "darkorange", "Bitcoin Price", "Day", "Normalized Price/Sentiment", "Normalized Price/Sentiment vs. Day - Bitcoin Subreddit", "normalized_day_bitcoin.png", sentiment_day_bitcoin_sorted_normalized, "lightblue", "Sentiment")

  # Plot of Bitcoin price vs. sentiment normalized (day)
  create_scatter_plot(sentiment_day_sorted_normalized, btc_day_sorted_normalized, "Sentiment", "Bitcoin Price", "Normalized Bitcoin Prices vs. Sentiment (day)", "normalized_regression_day.png")

  # Plot of Bitcoin price vs. sentiment normalized (day) - bitcoin subreddit only
  create_scatter_plot(sentiment_day_bitcoin_sorted_normalized, btc_day_sorted_normalized, "Sentiment", "Bitcoin Price", "Normalized Bitcoin Prices vs. Sentiment (day) - Bitcoin Subreddit Only", "normalized_regression_day_bitcoin.png")
