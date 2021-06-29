import pandas as pd

class CustomDF():
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def nullcount(self):
        print('This code is getting run')
        return self.df.isnull().sum().sum()

class CustomDF2():
    def __init__(self, filename, col):
        self.col = str(col)
        # super().__init__(pd.Series(filename), col)
        self.filename = str(filename)
        self.df = pd.read_csv(self.filename)

    def split_dates(self):
        self.dates_series = self.df[self.col]
        print('The series is display here', self.dates_series)
        self.dates_series_time = pd.to_datetime(self.dates_series, infer_datetime_format=True)
        print('The dates_series_time is display here', self.dates_series_time)
        self.month = self.dates_series_time.dt.month
        print('Shape for self.month',self.month.shape)
        self.day = self.dates_series_time.dt.day
        self.year = self.dates_series_time.dt.year
        self.frame = {"month": self.month, "day": self.day, "year": self.year}
        self.output = pd.DataFrame(self.frame)
        return self.output

if __name__ == '__main__':

    # instantiating object for the purpose of testing nullcount method in class CustomDF
    print('Running First line')
    # import file
    df_forest = CustomDF('ForestCover.csv')
    nullcountsvalue = df_forest.nullcount()
    print('nullcountsvalue: ', nullcountsvalue)
    #test method
    nullcounts = df_forest.nullcount()

    # instantiating another object for the purpose of testing split_dates method in class CustomDF2
    # import file
    df_dates = CustomDF2('Dates.csv', 'Date')
    print('Display df_dates', df_dates)
    # test method
    df_dates_changed = df_dates.split_dates()
    print('The new df with changed dates is: \n')
    print(df_dates_changed)
