# Day Trading

I'm starting this project to try to earn money by day trading three tickets: $AAPL, $NVDA, and $TSLA. I choose those tickets because many people in the Reddit group of `r/Daytrading` use those stocks. I think it's because each of them is a leading ticket for different markets: Consumer Electronics, Semiconductors, and Auto Manufacturers. Tickets with high levels of volume for trading probably are easier to predict.

## Dataset

Yahoo Finances provides an endpoint where I can extract important data about the tickets:

```
https://query1.finance.yahoo.com/v8/finance/chart/AAPL?region=US&lang=en-US&includePrePost=false&interval=5m&useYfid=true&range=1mo&corsDomain=finance.yahoo.com&.tsrc=finance
```

Currently, I just modified the URL to collect data every 5 minutes for a month of each ticket and manually copied the resulting data to get the initial dataset.
