from tabula import read_pdf as tb_pdf
import tabula
import pandas as pd
import numpy as np
from collections import defaultdict

# book = open_workbook('C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\Machine_Learning_Tool\\Test\\random.xls')

pdf = 'C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\Machine_Learning_Tool\\LDA Files\\RFQ_MV motor_SGP.0706-BSL.18.0420-F0.pdf'

df = tb_pdf(pdf, multiple_tables=True, pages='all', encoding='utf-8')

dataframe = pd.DataFrame()
for line in df:
    line.dropna(how='all', inplace=True)
    line.dropna(how='all', inplace=True, axis=1)
    lenght_columns = len(line.columns)
    list_columns = []
    for i in range(0, lenght_columns):
        list_columns.append(str(i))
    line.columns = list_columns
    if dataframe.empty:
        dataframe = line
    else:
        lenght_column = len(dataframe.columns) + 1
        for i in range(2, lenght_columns+1):
            lenght_column += 1
            for index_line, row_line in line.iterrows():
                for index_dataframe, row_dataframe in dataframe.iterrows():
                    if row_dataframe[0] == row_line[0]:
                        print(row_line[str(i)])
                        dataframe.loc[index_dataframe, str(lenght_column)] = row_line[str(i)]


        '''column_numbers = [x for x in range(line.shape[1])]  # list of columns' integer indices
        column_numbers.remove(0)
        column_numbers.remove(1)
        line = line.iloc[:, column_numbers]

        frames = [dataframe, line]
        dataframe = pd.concat(frames, axis=1)'''

'''dataframe = dataframe.rename(columns={np.nan: 'initial_value'})
for column_name, column_value in dataframe.iteritems():
    #print(column_name)
    dataframe.dropna(subset=[column_name], inplace=True)'''

dataframe.to_excel(
            'C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\Machine_Learning_Tool\\Test\\newly_read_pdf_file_RFQ_MV motor_SGP.0706-BSL.18.0420-F0.xlsx',
            index=False)
