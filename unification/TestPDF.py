from tabula import read_pdf as tb_pdf
import tabula
import pandas as pd
from collections import defaultdict
from numpy import *
import xlsxwriter

# book = open_workbook('C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\Machine_Learning_Tool\\Test\\random.xls')

pdf = 'C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\Machine_Learning_Tool\\LDA Files\\RFQ_MV motor_SGP.0706-BSL.18.0420-F0.pdf'

df = tb_pdf(pdf, multiple_tables=True, pages='all', encoding='utf-8')

# print(type(df))

dictOfWords = {i: df[i] for i in range(0, len(df))}

# print(dictOfWords)
datafr = pd.DataFrame(df)

name = 'C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\' \
       'Machine_Learning_Tool\\Test\\random.xlsx'

# pd.DataFrame(df).to_excel(name, header=False, index=False)

tmp = df
#print(tmp)
df = pd.DataFrame(df).T.set_index(0).T
#print(df.head())

tmp = [incom for incom in tmp if str(incom) != 'nan']
measures = tmp[0]
print(measures)

try:
    d_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            d_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    save_report = pd.DataFrame.from_dict(d_class_data).T
    save_report.index.name = 'Labels'

    save_report.to_excel(name, index=True)
except IndexError:
    message = 'Cannot be created'


#datafr.to_excel('C:\\Users\\z003xe7x\\Desktop\\Unification_Service_Machine_Learning_Application\\Machine_Learning_Tool\\Test\\newly_read_pdf_file_RFQ_MV motor_SGP.0706-BSL.18.0420-F0.xlsx', index=False)