# pandas
#This is an exercise I made as a preparation for an interview for a position in a financial data. The company's main specific requirement was knowledge of pandas module, so I made as much things in pandas a possible.

- Data

It is a CSV of investments in startups from keggle (https://www.kaggle.com/justinas/startup-investments) but it was slightly adapted, thus I provide the adapted CSV here. Each row is a startup with columns being data about them such as the city where it is located, date of funding, different financial characteristics (how many venture capital was injected, how many funding rounds was there), status (is it still operating or not), etc... Due to computational limitations, only a subset of data was used (15000 rows, about 38% of all rows)

- Goal

The goal was to create a model that from all the characteristics of startup predicts how much funding this startup will receive

- Method

I cleaned the data, did a little bit of data wrangling and created a random forest regressor model that best predicts the funding the startup will receive. The success of prediction is expressed by mean of error of prediction in absolute percentages, and is given by summatory of:
abs(((predicted_funding - real_funding)/real funding)*100)

- Organization

The code is organized in 4 blocks:
1. Data summary, description, data wrangling using pandas module
2. Basic data processing and preparation compatible with ML
3. Improving model by more advanced data processing and improvement
4. Improving model by tuning the ML technique itself

- Results

With only basic data preparation, the mean error of prediction was 11.55%
Addition data processing innitially decreased the prediction potential to mean error of 12.65%
After further cleaning of data and removing non-important and redundant variables, the mean error decreased to 11.44%
After tuning the parameters of the predictor, the final mean error was decreased to 9.96%

- Summary

Someone who uses this predictor, can expect prediction with accuracy of -+10%. Eg: you want to create a startup, you introduce parameters of your startup and model outputs you can expect a funding of 1000000USD. It means you can confidently believe that you will get funding of between 900000USD and 1100000USD. How confident can you be in this number can be also acessed by the standard deviation of errors, also given by the program. Whether the model underestimates funding more often than overestimates or vice-versa can be acessed by normal distribution curve of errors also given by the programms. All of that is of course approximations and will be improved if the whole dataset will be used instead of subset that I used. The program can be improved further. I know how. Hire me and I'll do it lol
