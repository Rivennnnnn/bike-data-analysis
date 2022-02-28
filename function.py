import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from pyecharts.charts import Line,Page,Bar,Timeline
import pyecharts.options as opts
from pyecharts.charts import Bar
def preprocess():
    # 数据预处理
    train_WithoutOutliers = train[np.abs(train['count'] - train['count'].mean()) <= (3 * train['count'].std())]
    train_WithoutOutliers.shape
    train_WithoutOutliers['count'].describe()

    # 对数变换
    yLabels = train_WithoutOutliers['count']
    yLabels_log = np.log(yLabels)
    #sns.distplot(yLabels_log)

    # 合并数据集
    Bike_data = pd.concat([train_WithoutOutliers, test], ignore_index=True)
    # Bike_data.shape
    Bike_data.head()

    from datetime import datetime
    Bike_data['date'] = Bike_data.datetime.apply(lambda c: c.split()[0])
    Bike_data['hour'] = Bike_data.datetime.apply(lambda c: c.split()[1].split(':')[0]).astype('int')
    Bike_data['year'] = Bike_data.datetime.apply(lambda c: c.split()[0].split('-')[0]).astype('int')
    Bike_data['month'] = Bike_data.datetime.apply(lambda c: c.split()[0].split('-')[1]).astype('int')
    Bike_data['weekday'] = Bike_data.date.apply(lambda c: datetime.strptime(c, '%Y-%m-%d').isoweekday())
    Bike_data.head()

    from sklearn.ensemble import RandomForestRegressor

    Bike_data["windspeed_rfr"] = Bike_data["windspeed"]
    # 将数据分成风速等于0和不等于两部分
    dataWind0 = Bike_data[Bike_data["windspeed_rfr"] == 0]
    dataWindNot0 = Bike_data[Bike_data["windspeed_rfr"] != 0]
    # 选定模型
    rfModel_wind = RandomForestRegressor(n_estimators=1000, random_state=42)
    # 选定特征值
    windColumns = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]
    # 将风速不等于0的数据作为训练集，fit到RandomForestRegressor之中
    rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed_rfr"])
    # 通过训练好的模型预测风速
    wind0Values = rfModel_wind.predict(X=dataWind0[windColumns])
    # 将预测的风速填充到风速为零的数据中
    dataWind0.loc[:, "windspeed_rfr"] = wind0Values
    # 连接两部分数据
    Bike_data = dataWindNot0.append(dataWind0)
    Bike_data.reset_index(inplace=True)
    Bike_data.drop('index', inplace=True, axis=1)
    #Bike_data.to_csv("Bike_data.csv",index=False)
    return Bike_data


def predict(Bike_data):
    #Bike_data = preprocess()
    from sklearn.ensemble import RandomForestRegressor
    # 数据预测
    dummies_month = pd.get_dummies(Bike_data['month'], prefix='month')
    dummies_season = pd.get_dummies(Bike_data['season'], prefix='season')
    dummies_weather = pd.get_dummies(Bike_data['weather'], prefix='weather')
    dummies_year = pd.get_dummies(Bike_data['year'], prefix='year')
    # 把5个新的DF和原来的表连接起来
    Bike_data = pd.concat([Bike_data, dummies_month, dummies_season, dummies_weather, dummies_year], axis=1)
    Bike_data.to_csv('test.csv', index=False)

    dataTrain = Bike_data[pd.notnull(Bike_data['count'])]
    dataTest = Bike_data[~pd.notnull(Bike_data['count'])].sort_values(by=['datetime'])
    datetimecol = dataTest['datetime']
    yLabels = dataTrain['count']
    yLabels_log = np.log(yLabels)

    dropFeatures = ['casual', 'count', 'datetime', 'date', 'registered', 'month', 'season', 'weather', 'year',
                    'holiday', 'workingday', ]
    dataTrain = dataTrain.drop(dropFeatures, axis=1)
    dataTest = dataTest.drop(dropFeatures, axis=1)

    rfModel = RandomForestRegressor(n_estimators=1000, random_state=42)
    rfModel.fit(dataTrain, yLabels_log)
    preds = rfModel.predict(X=dataTrain)

    predsTest = rfModel.predict(X=dataTest)
    submission = pd.DataFrame({'datetime': datetimecol, 'count': [max(0, x) for x in np.exp(predsTest)]})
    submission = submission.drop(submission.tail(6236).index,inplace=False)

    submission['day'] = submission.datetime.apply(lambda c: c.split()[0].split('-')[2]).astype('int')
    submission['hour'] = submission.datetime.apply(lambda c: c.split()[1].split(':')[0]).astype('int')
    tl = Timeline()
    for day in range(20,32):
        temp = submission.loc[submission["day"]==day]
        bar = (
            Bar(init_opts=opts.InitOpts(width="1600px",height="800px"))
            .add_xaxis(
                xaxis_data=list(temp["hour"])
            )
            .add_yaxis(
                series_name="count",
                y_axis=list(temp["count"])
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="{}日预测额".format(day))
            )
        )
        tl.add(bar,day)
    tl.render("./html/predict.html")

def hour(Bike_data):
    #Bike_data = preprocess()
    workingday_df = Bike_data[Bike_data['workingday'] == 1]
    workingday_df = workingday_df.groupby(['hour'], as_index=True).agg({
        'casual': 'mean', 'registered': 'mean', 'count': 'mean'})
    nworkingday_df = Bike_data[Bike_data['workingday'] == 0]
    nworkingday_df = nworkingday_df.groupby(['hour'], as_index=True).agg({
        'casual': 'mean', 'registered': 'mean', 'count': 'mean'})
    nworkingday_chart = (   Line(init_opts=opts.InitOpts(width="1600px",height="800px"))
        .add_xaxis(
            xaxis_data=nworkingday_df.index
        )
        .add_yaxis(
            series_name="casual",
            y_axis=nworkingday_df['casual'],
            is_symbol_show=False,
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average",name="平均值")]
            )
        )
        .add_yaxis(
            series_name="registered",
            y_axis=nworkingday_df['registered'],
            is_symbol_show=False,
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average",name="平均值")]
            )
        )
        .add_yaxis(
            series_name="count",
            y_axis=nworkingday_df['count'],
            is_symbol_show=False,
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average",name="平均值")]
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="The average number of rentals initiated per hour in the nonworking days"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
    )
    workingday_chart = (   Line(init_opts=opts.InitOpts(width="1600px",height="800px"))
        .add_xaxis(
            xaxis_data=workingday_df.index
        )
        .add_yaxis(
            series_name="casual",
            y_axis=workingday_df['casual'],
            is_symbol_show=False,
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average",name="平均值")]
            )
        )
        .add_yaxis(
            series_name="registered",
            y_axis=workingday_df['registered'],
            is_symbol_show=False,
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average",name="平均值")]
            )
        )
        .add_yaxis(
            series_name="count",
            y_axis=workingday_df['count'],
            is_symbol_show=False,
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average",name="平均值")]
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="The average number of rentals initiated per hour in the working days"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
    )
    (Page()
     .add(nworkingday_chart)
     .add(workingday_chart)
     .render("./html/hour.html")
     )
def temp(Bike_data):
    #Bike_data = preprocess()
    # 数据按照小时统计展示起来太麻烦，希望能够按天汇总取一天的气温中位数
    temp_df = Bike_data.groupby(['date', 'weekday'], as_index=False).agg(
        {'year': 'mean', 'month': 'mean', 'temp': 'median'})
    # 由于测试数据集中没有租赁信息，会导致折线图有断裂，所以将缺失的数据丢弃
    temp_df.dropna(axis=0, how='any', inplace=True)
    # 预计按天统计的波动仍然很大，再按月按日取平均值
    temp_month = temp_df.groupby(['year', 'month'], as_index=False).agg({'weekday': 'min', 'temp': 'median'})
    # 将按天求和统计数据的日期转换成datetime格式
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    # 将按月统计数据设置一列时间序列
    temp_month.rename(columns={'weekday': 'day'}, inplace=True)
    temp_month['date'] = pd.to_datetime(temp_month[['year', 'month', 'day']])
    (Line(init_opts=opts.InitOpts(width="2000px",height="800px"))
        .add_xaxis(
            xaxis_data=temp_df['date']
        )
        .add_yaxis(
            series_name="Daily average",
            y_axis=temp_df['temp'],
            is_symbol_show=False
        )
        .add_xaxis(
            xaxis_data=temp_month['date']
        )
        .add_yaxis(
            series_name='Monthly average',
            y_axis=temp_month['temp'],
            is_symbol_show=False
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Change trend of average temperature per day in two years"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
        .render("./html/temp.html")
    )
def humidity(Bike_data):
    #Bike_data = preprocess()
    humidity_df = Bike_data.groupby('date', as_index=False).agg({'humidity': 'mean'})
    humidity_df['date'] = pd.to_datetime(humidity_df['date'])
    # 将日期设置为时间索引
    humidity_df = humidity_df.set_index('date')
    humidity_month = Bike_data.groupby(['year', 'month'], as_index=False).agg({'weekday': 'min', 'humidity': 'mean'})
    humidity_month.rename(columns={'weekday': 'day'}, inplace=True)
    humidity_month['date'] = pd.to_datetime(humidity_month[['year', 'month', 'day']])
    (Line(init_opts=opts.InitOpts(width="2000px",height="800px"))
        .add_xaxis(
            xaxis_data=humidity_df.index
        )
        .add_yaxis(
            series_name="Daily average",
            y_axis=humidity_df['humidity'],
            is_symbol_show=False
        )
        .add_xaxis(
            xaxis_data=humidity_month['date']
        )
        .add_yaxis(
            series_name='Monthly average',
            y_axis=humidity_month['humidity'],
            is_symbol_show=False
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Change trend of average humidity per day in two years"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
        .render("./html/humidity.html")
    )
def yearsmonths(Bike_data):
    #Bike_data = preprocess()
    # 数据按小时统计展示起来太麻烦，希望能够按天汇总
    count_df = Bike_data.groupby(['date', 'weekday'], as_index=False).agg(
        {'year': 'mean', 'month': 'mean', 'casual': 'sum', 'registered': 'sum', 'count': 'sum'})
    # 由于测试数据集中没有租赁信息，会导致折线图有断裂，所以将缺失的数据丢弃
    count_df.dropna(axis=0, how='any', inplace=True)
    # 预计按天统计的波动仍然很大，再按月取日平均值
    count_month = count_df.groupby(['year', 'month'], as_index=False).agg(
        {'weekday': 'min', 'casual': 'mean', 'registered': 'mean', 'count': 'mean'})
    # 将按天求和统计数据的日期转换成datetime格式
    count_df['date'] = pd.to_datetime(count_df['date'])
    # 将按月统计数据设置一列时间序列
    count_month.rename(columns={'weekday': 'day'}, inplace=True)
    count_month['date'] = pd.to_datetime(count_month[['year', 'month', 'day']])
    (Line(init_opts=opts.InitOpts(width="1800px",height="600px"))
        .add_xaxis(
            xaxis_data=count_df['date']
        )
        .add_yaxis(
            series_name="Daily average",
            y_axis=count_df['count'],
            is_symbol_show=False
        )
        .add_xaxis(
            xaxis_data=count_month['date']
        )
        .add_yaxis(
            series_name='Monthly average',
            y_axis=count_month['count'],
            is_symbol_show=False
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Change trend of average number of rentals initiated per day in two years"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
        .render("./html/yearsmonths.html")
    )
def season(Bike_data):
    #Bike_data = preprocess()
    day_df = Bike_data.groupby('date').agg(
        {'year': 'mean', 'season': 'mean', 'casual': 'sum', 'registered': 'sum', 'count': 'sum', 'temp': 'mean',
         'atemp': 'mean'})
    season_df = day_df.groupby(['year', 'season'], as_index=True).agg(
        {'casual': 'mean', 'registered': 'mean', 'count': 'mean'})
    temp_df = day_df.groupby(['year', 'season'], as_index=True).agg({'temp': 'mean', 'atemp': 'mean'})
    temp_chart = (   Line(init_opts=opts.InitOpts(width="1600px",height="800px"))
        .add_xaxis(
            xaxis_data=temp_df.index
        )
        .add_yaxis(
            series_name="temp",
            y_axis=temp_df['temp'],
            is_symbol_show=False
        )
        .add_yaxis(
            series_name="atemp",
            y_axis=temp_df['atemp'],
            is_symbol_show=False
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="The trend of average temperature per day changes with season"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
    )
    season_chart = (   Line(init_opts=opts.InitOpts(width="1600px",height="800px"))
        .add_xaxis(
            xaxis_data=season_df.index
        )
        .add_yaxis(
            series_name="casual",
            y_axis=season_df['casual'],
            is_symbol_show=False
        )
        .add_yaxis(
            series_name="registered",
            y_axis=season_df['registered'],
            is_symbol_show=False
        )
        .add_yaxis(
            series_name="count",
            y_axis=season_df['count'],
            is_symbol_show=False
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="The trend of average temperature per day changes with seanson"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False)
        )
    )
    (Page()
     .add(temp_chart)
     .add(season_chart)
     .render("./html/season.html")
     )
def weather(Bike_data):
    #Bike_data = preprocess()
    count_weather = Bike_data.groupby('weather')
    count_weather[['casual', 'registered', 'count']].count()
    weather_df = Bike_data.groupby('weather', as_index=True).agg({'casual': 'mean', 'registered': 'mean'})
    from pyecharts.charts import Bar
    (
        Bar()
            .add_xaxis(
                xaxis_data=list(weather_df.index)
            )
            .add_yaxis(
                series_name="casual",
                y_axis=list(weather_df["casual"]),
                stack="stack1"
            )
            .add_yaxis(
                series_name="registered",
                y_axis=list(weather_df["registered"]),
                stack="stack1"
            )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="Average number of rentals initiated per hour in different wearther")
            )
            .render("./html/weather.html")
    )
def windspeed(Bike_data):
    #Bike_data = preprocess()
    windspeed_df = Bike_data.groupby('date', as_index=False).agg({'windspeed_rfr': 'mean'})
    windspeed_df['date'] = pd.to_datetime(windspeed_df['date'])
    # 将日期设置为时间索引
    windspeed_df = windspeed_df.set_index('date')
    windspeed_month = Bike_data.groupby(['year', 'month'], as_index=False).agg(
        {'weekday': 'min', 'windspeed_rfr': 'mean'})
    windspeed_month.rename(columns={'weekday': 'day'}, inplace=True)
    windspeed_month['date'] = pd.to_datetime(windspeed_month[['year', 'month', 'day']])

    windspeed_rentals = Bike_data.groupby(['windspeed'], as_index=True).agg(
        {'casual': 'max', 'registered': 'max', 'count': 'max'})

    #windspeed_rentals.plot(title='Max number of rentals initiated per hour in different windspeed ')
    #plt.show()

    (
        Line()
            .add_xaxis(
                xaxis_data=windspeed_rentals.index
            )
            .add_yaxis(
                series_name="总量",
                y_axis=windspeed_rentals["count"]
            )
            .add_yaxis(
                series_name="游客",
                y_axis=windspeed_rentals["casual"]
            )
            .add_yaxis(
                series_name="会员",
                y_axis=windspeed_rentals["registered"]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="风速与出行量x")
           )
            .render("./html/wind.html")
    )
def datetimer(Bike_data):
    #Bike_data = preprocess()
    day_df = Bike_data.groupby(['date'], as_index=False).agg(
        {'casual': 'sum', 'registered': 'sum', 'count': 'sum', 'workingday': 'mean', 'weekday': 'mean',
         'holiday': 'mean', 'year': 'mean'})
    day_df.head()
    number_pei = day_df[['casual', 'registered']].mean()

    workingday_df = day_df.groupby(['workingday'], as_index=True).agg({'casual': 'mean', 'registered': 'mean'})

    weekday_df = day_df.groupby(['weekday'], as_index=True).agg({'casual': 'mean', 'registered': 'mean'})
    #weekday_df.plot.bar(stacked=True, title='Average number of rentals initiated per day by weekday')
    #plt.show()

    weekday_bar = (
        Bar(init_opts=opts.InitOpts(width="1600px",height="800px"))
            .add_xaxis(
                xaxis_data=list(weekday_df.index)
            )
            .add_yaxis(
                series_name="casual",
                y_axis=list(weekday_df["casual"]),
                stack="stack1"
            )
            .add_yaxis(
                series_name="registered",
                y_axis=list(weekday_df["registered"]),
                stack="stack1"
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Average number of rentals initiated per day by weekday")
            )
    )


    holiday_df = day_df.groupby('holiday', as_index=True).agg({'casual': 'mean', 'registered': 'mean'})
   # holiday_df.plot.bar(stacked=True, title='Average number of rentals initiated per day by holiday or not')
    #plt.show()

    holiday_bar = (
        Bar(init_opts=opts.InitOpts(width="1600px",height="800px"))
            .add_xaxis(
                xaxis_data=list(holiday_df.index)
            )
            .add_yaxis(
                series_name="casual",
                y_axis=list(holiday_df["casual"]),
                stack="stack1"
            )
            .add_yaxis(
                series_name="registered",
                y_axis=list(holiday_df["registered"]),
                stack="stack1"
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Average number of rentals initiated per day by holiday or not")
            )
    )
    (
        Page()
        .add(weekday_bar)
        .add(holiday_bar)
        .render("./html/datetimer.html")
    )
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    train = pd.read_csv('E:/train.csv')
    train.info()
    test = pd.read_csv('E:/test.csv')
    test.info()
    train.describe()
    Bike_data = preprocess()
    if not os.path.exists("./html"):
        os.mkdir("./html")
    hour(Bike_data)
    temp(Bike_data)
    humidity(Bike_data)
    yearsmonths(Bike_data)
    season(Bike_data)
    weather(Bike_data)
    windspeed(Bike_data)
    datetimer(Bike_data)
    predict(Bike_data)


