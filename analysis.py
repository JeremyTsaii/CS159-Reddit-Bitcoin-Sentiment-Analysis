import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats

# Create plot with given data and labels
# Used for plots of data (Bitcoin prices or sentiments) over time
def create_plot(xData1, yData1, color1, legend1, xLabel, yLabel, title, yData2=None, color2=None, legend2=None):
  plt.plot(xData1, yData1, color = color1, label=legend1)
  if yData2 and color2 and legend2:
    plt.plot(xData1, yData2, color=color2, label=legend2)
  
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.title(title)
  plt.legend()
  plt.show()

# Create scatter plot with given data and labels
# Used for correlation plots between sentiment and Bitcoin prices
def create_scatter_plot(xData, yData, xLabel, yLabel, title):
  # Create linear regression
  slope, intercept, r_value, p_value, std_err = stats.linregress(xData, yData)
  regression = np.linspace(-1,1) * slope + intercept
  print(f'p-value: {p_value}')
  print(f'R^2: {r_value * r_value}')
  print(f'slope: {slope}')
  print(f'standard error: {std_err}')

  plt.scatter(xData, yData)
  plt.plot(np.linspace(-1,1), regression)
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.title(title)
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

if __name__ == "__main__":
  analyzer = SentimentIntensityAnalyzer()

  x = [1, 2, 3, 4]
  y = [
    get_sentiment(analyzer, "Wow this movie was mediocre"),
    get_sentiment(analyzer, "Congratulations, great job!"),
    get_sentiment(analyzer, "Omg I hate you so much ughhhhh"),
    get_sentiment(analyzer, "This is the worst day of my life")
  ]

  y2 = [.2, .2, .2, .2]
  create_plot(x, y, "darkorange", "Sentiment", "X Label", "Y Label", "Plot Title", y2, "lightblue", "Random Stuff")

  x2 = [0.5,-0.2,-0.8,0.2]
  create_scatter_plot(x2, y, "X Label", "Y Label", "Plot Title")

