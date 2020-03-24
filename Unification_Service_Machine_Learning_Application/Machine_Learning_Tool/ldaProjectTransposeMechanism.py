import random
import pandas as pd
import sys, os


class DoTranspose:
    def do_transposer(self, mappingfilepath, dataframefilepath, dataframe_sheet_name):
        self.mappingfilepath = mappingfilepath
        self.dataframefilepath = dataframefilepath
        self.dataframe_sheet_name = dataframe_sheet_name

        dataframe = pd.read_excel(dataframefilepath, sheet_name=dataframe_sheet_name, encoding='utf-8', header=0)
        mappingfile = pd.read_excel(mappingfilepath, encoding='utf-8', header=0)
        dataframe = dataframe.fillna('No_Value')

        no_of_rows_matched = 0
        start_index = -1
        matches_in_a_row_final = 0



        for indexmap, rowmap in mappingfile.iterrows():

            for index, row in dataframe.iterrows():
                matches_in_a_row = 0
                for r in row:
                    row_match = 0
                    if r is str(r):
                        r = r.replace('(', '[')
                        r = r.replace(')', ']')
                        r = str(r)
                        word_match = 0
                        for stringr in r.split():
                            for word in rowmap.Initial_Value.split():
                                if stringr == word:
                                    word_match += 1
                                    if start_index == -1:
                                        start_index = index
                                    #print('string r: ', str(r), 'word: ', word, 'word match: ', word_match)
                            if word_match > 1:
                                print('word_match: ', word_match)
                                row_match = row_match + 1
                        if row_match > 1:
                            print('row_match: ', row_match)
                            matches_in_a_row = matches_in_a_row + 1
                    if matches_in_a_row > 1:
                        print('matches_in_a_row: ', matches_in_a_row)
                        no_of_rows_matched = no_of_rows_matched + 1
                        matches_in_a_row_final = matches_in_a_row
        dataframe_transposed = pd.DataFrame

        print('matches_in_a_row_final: ', matches_in_a_row_final)
        print('no_of_rows_matched: ', no_of_rows_matched)
        if matches_in_a_row_final > 2:
            dataframe = dataframe.drop(dataframe.index[:start_index])
            dataframe = dataframe.reset_index(drop=True)

            dataframe = dataframe.astype(str)
            dataframe_transposed = dataframe.T

            dataframe_transposed.insert(loc=0, column="Initial_Value", value="")

            # dataframe_transposed["Initial_Value"] = [' '.join(row) for row in dataframe_transposed[dataframe_transposed.columns[
            #                                                                                        1:no_of_rows_matched]].values]

            for (columnName, columnData) in dataframe_transposed.iteritems():
                if dataframe_transposed.columns.get_loc(columnName) <= no_of_rows_matched:
                    dataframe_transposed['Initial_Value'] = dataframe_transposed['Initial_Value'] + " " + dataframe_transposed[columnName]

            no_of_rows_matched = no_of_rows_matched + 2
            list_to_delete = []
            for row in range(1, no_of_rows_matched):
                list_to_delete.append(row)

            dataframe_transposed.drop(dataframe_transposed.columns[list_to_delete], axis=1, inplace=True)
            dataframe_transposed['Initial_Value'] = dataframe_transposed['Initial_Value'].str.replace('No_Value', '')
            dataframe_transposed['Initial_Value'] = dataframe_transposed['Initial_Value'].str.replace('\n', '')
            dataframe_transposed['Initial_Value'] = dataframe_transposed['Initial_Value'].map(lambda x: x.strip())
            dataframe_transposed.reset_index(drop=True, inplace=True)

            index_list = []
            for index, row in dataframe_transposed.iterrows():
                flag = 0
                for indexmap, rowmap in mappingfile.iterrows():
                    if str(row.Initial_Value) in rowmap.Initial_Value:
                        flag = 1
                        break
                    elif rowmap.Initial_Value in str(row.Initial_Value):
                        flag = 1
                        break
                    elif str(row.Initial_Value) in rowmap.Initial_Value:
                        row.Initial_Value = row.Initial_Value.replace('(', '[')
                        row.Initial_Value = row.Initial_Value.replace(')', ']')
                        flag = 1
                        break
                    elif rowmap.Initial_Value in str(row.Initial_Value):
                        row.Initial_Value = row.Initial_Value.replace('(', '[')
                        row.Initial_Value = row.Initial_Value.replace(')', ']')
                        flag = 1
                        break
                if flag == 0:
                    index_list.append(index)

            dataframe_transposed = dataframe_transposed.drop(dataframe_transposed.index[index_list])
            dataframe_transposed.reset_index(drop=True, inplace=True)

            columnslist = list(dataframe_transposed)
            mylist = ['False', 'No_Value', '0']
            pattern = '|'.join(mylist)
            total_count = 0
            found_count = 0
            deleted_col_list = []

            dataframe_transposed.columns = dataframe_transposed.columns.astype(str)
            #dataframe_transposed.to_excel('D:\\shared\\Machine Learning Tool\\LDA Files\\testdatatransfinal.xlsx',
            # index=False)
            for (columnName, columnData) in dataframe_transposed.iteritems():
                total_count = 0
                found_count = 0
                for values in columnData.values:
                    if values in pattern:
                        found_count += 1
                    total_count += 1
                if found_count / total_count * 100 > 80:
                    #print('Colunm Name : ', columnName)
                    deleted_col_list.append(dataframe_transposed.columns.get_loc(columnName))

            dataframe_transposed.drop(dataframe_transposed.columns[deleted_col_list], axis=1, inplace=True)
        # dataframe_transposed.to_excel('D:\\shared\\Machine Learning Tool\\LDA Files\\testdatatransfinalfile.xlsx',
        #                               index=False)
            path_to_save = os.getcwd() + '/Data Sets/Cleaned_File.xlsx'
            dataframe_transposed.to_excel(path_to_save, index=False)

        flag = 0
        if dataframe_transposed.empty:
            flag = 0
        else:
            flag = 1
        return dataframe_transposed


def main():
    trans = DoTranspose()
    dataframefilepath = 'C:\\Users\\Z003XE7X\Documents\\Unification Service Machine Learning Application\\Machine ' \
                        'Learning Tool\\LDA Files\\Kopie von Motor Enquiry (2).xlsx'#Kopie von MV Motor Table -
    # DES-18-3877 - REV.2.xlsx'
    dataframe_sheet_name = 'Motors'#'MV EM'
    mappingfilepath = 'C:\\Users\\Z003XE7X\Documents\\Unification Service Machine Learning Application\\Machine ' \
                        'Learning Tool\\LDA Files\\Mapping_Technical_Data.xlsx'
    trans.do_transposer(mappingfilepath, dataframefilepath, dataframe_sheet_name)

if __name__ == '__main__':
    main()