from mysklearn import myutils

import copy 
import csv
from tabulate import tabulate


class MyPyTable:
    """
    purpose: represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
        
    note: there are N rows by M columns.
    """


    def __init__(self, column_names=None, data=None):
        """
        Purpose: 
            The initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        
        self.column_names = copy.deepcopy(column_names)
        
        if data is None:
            data = []
        
        self.data = copy.deepcopy(data)


    def pretty_print(self):
        """
        Purpose: 
            prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """
        Purpose:
            computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        N = len(self.data) # number of rows
        M = len(self.column_names) # number of columns. 

        return (N, M) 


    def get_column(self, col_identifier, include_missing_values=True):
        """
        Purpose: 
            extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int for a column index
            include_missing_values (bool): True if missing values ("NA") should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """

        # ensure col_identifier is valid. If not, raise ValueError.
        if type(col_identifier) is str:
            if col_identifier not in self.column_names:
                raise ValueError(f"{col_identifier} is an invalid column identifier")

            col_idx = self.column_names.index(col_identifier)
        
        elif type(col_identifier) is int:
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError(f"{col_identifier} is an invalid column identifier")
            
            col_idx = col_identifier
        
        else:
            raise ValueError(f"{col_identifier} is an invalid column identifier (not type str or int)")
        
        col_values = []

        # extract all rows from the column, specified by col_idx
        for row in self.data:
            row_value = row[col_idx]

            if include_missing_values:
                # if row_value != "NA":
                    col_values.append(row_value)
            else:
                if row_value == "NA":
                    continue

        
        return col_values


    def convert_to_numeric(self):
        """
        Purpose:
            Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """

        for row in self.data:
            for row_idx, row_val in enumerate(row):
                
                # try to convert the value, row_val, to a float.
                try: 
                    row[row_idx] = float(row_val)
                
                except ValueError:
                    # print("Value cannot be converted to numeric type (float).")
                    continue
            

    def drop_rows(self, row_indexes_to_drop):
        """
        Purpose: 
            Removes rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        
        # sort row indices to drop in descending order. Do this so we start deleting from back of list and avoid mixmatch of indices due to shifting.
        row_indexes_to_drop.sort(reverse=True)

        # iterate through each item in the list of indices to remove from the data table 
        for idx in row_indexes_to_drop:
            del self.data[idx] # remove row from data table corresponding to the index to remove.


    def load_from_file(self, filename):
        """
        Purpose: 
            Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, "r") as infile:
            contents = csv.reader(infile)
            rows = list(contents)

        # assume first row is header
        self.column_names = rows[0]
        self.data = rows[1:]

        self.convert_to_numeric()
        
        return self



    def save_to_file(self, filename):
        """
        Purpose: 
            Save table column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, "w") as outfile:
            writer = csv.writer(outfile)

            writer.writerow(self.column_names) # write header to csv file.
            writer.writerows(self.data) # write data to csv file.


    def find_duplicates(self, key_column_names):
        """
        Purpose: 
            Returns a list of indexes representing duplicate rows.
            Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """

        # initialize a list to store indices (ints) of duplicate rows.
        duplicates = []

        # initialize a set to store the key of seen rows 
        seen = set()

        # find indices of all key_col's. 
        key_col_indices = [self.column_names.index(key) for key in key_column_names]

        # use enumerate so we can access index and value while iterating through the data.
        for row_idx, row_val in enumerate(self.data):

            # create a composite key for comparing rows rather than comparing key columns separately.
            full_key = tuple(row_val[key] for key in key_col_indices)

            # if a row we've iterated over has the same composite key as this row, append the row to the duplicates list.
            if full_key in seen:
                duplicates.append(row_idx)
                print("full key of duplicate: ", full_key)
            else:
                seen.add(full_key)

        print("number of duplicates found: ", len(duplicates))
        return duplicates

            
    def remove_rows_with_missing_values(self):
        """
        Purpose: 
            Remove rows from the table data that contain a missing value ("NA").
        """

        rows_with_missing = []

        # iterate through data to get the row_idx and corresponding row values of each row.
        for row_idx, row_vals in enumerate(self.data):
            
            # if the row contains a missing value, append the row index to the list.
            if "NA" in row_vals:
                rows_with_missing.append(row_idx)


        # remove rows from table using the row idx list 
        for row_idx in sorted(rows_with_missing, reverse=True):
            del self.data[row_idx]


    def replace_missing_values_with_column_average(self, col_name):
        """
        Purpose: 
            For columns with continuous data, fill missing values in a column by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """

        # find the index of given column. 
        col_idx = self.column_names.index(col_name)

        row_values = []
        
        # find all non-missing row values in the given colum. 
        for row in self.data:
            value = row[col_idx]
            if value != "NA":
                row_values.append(float(value))

        # take row avg.
        avg = sum(row_values) / len(row_values)

        # fill in missing values
        for row in self.data:
            value = row[col_idx]
            if value == "NA":
                row[col_idx] = avg


   
    def compute_summary_statistics(self, col_names):
        """
        Purpose: 
            Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """

        header_names = ["attribute", "min", "max", "mid", "avg", "median"]

        # initialize a new MyPyTable object.
        summary_table = MyPyTable(header_names, [])
        
        print("number of columns to compute statistics for: ", len(col_names))

        # iterate through all columns.
        for col in col_names:
            col_idx = self.column_names.index(col)

            row_values = []

            # get all values in the current column. 
            for row in self.data:
                value = row[col_idx]
                if value != "NA": # ignore missing values.
                    row_values.append(float(value))
            
            # handle columns with no values in them.
            if not row_values:
                col_min = None
                col_max = None
                col_mid = None
                col_avg = None
                col_median = None
                continue

            # assign values to variables for current column in table.
            col_min = min(row_values)
            col_max = max(row_values)
            col_mid = (col_min + col_max) / 2
            col_avg = sum(row_values) / len(row_values)
            
            # find median...
            row_values.sort()

            if len(row_values) % 2 == 0: # even case.
                m1 = row_values[len(row_values)//2] # middle element.
                m2 = row_values[len(row_values)//2 - 1] # element before middle element.
                col_median = (m1 + m2) / 2 # avg of two middle elements.
            else: # odd case.
                col_median = row_values[len(row_values)//2]

            # append column summary values to the table.
            summary_table.data.append([col, col_min, col_max, col_mid, col_avg, col_median])

        return summary_table


    def perform_inner_join(self, other_table, key_column_names):
        """
        Purpose: 
            Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.

        Note:
            key columns are columns that contain UNIQUE identifiers to label each row in a dataset (i.e., "order_num", "orderID")
            non-key columns are columns that are just descriptive attributes (i.e., "product_name", "order_date")
        """

        # find indices of key columns in each table.
        key_col_indices = [self.column_names.index(col) for col in key_column_names]
        other_key_col_indices = [other_table.column_names.index(col) for col in key_column_names]

        # find indices of non-key columns in each table.
        non_key_col_indices = []
        non_other_key_col_indices = []

        for col in self.column_names:
            col_idx = self.column_names.index(col)
            if col_idx not in key_col_indices:
                non_key_col_indices.append(col_idx)
        
        for col in other_table.column_names:
            col_idx = other_table.column_names.index(col)
            if col_idx not in other_key_col_indices:
                non_other_key_col_indices.append(col_idx)

        # get column names for the combined table.
        self_col_names = self.column_names
        other_col_names = [other_table.column_names[col] for col in non_other_key_col_indices]
        combined_col_names = self_col_names + other_col_names

        # create a new MyPyTable object to fill with joined tables.
        combined_table = MyPyTable(combined_col_names, []) 
                
        # begin inner join of table values.        
        for row1 in self.data:
            combined_key1 = tuple(row1[key] for key in key_col_indices) # get values in all key cols in the self table.

            for row2 in other_table.data:
                combined_key2 = tuple(row2[key] for key in other_key_col_indices) # get values in all key cols in the other table.

                if combined_key1 == combined_key2:
                    extra_row2_cols = [row2[key] for key in non_other_key_col_indices]
                    combined_table.data.append(row1 + extra_row2_cols)
    
        return combined_table


    def perform_full_outer_join(self, other_table, key_column_names):
        """
        Purpose: 
            Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """

        # initialize a set to store 
        matched = set()

        # find indices of key columns in each table.
        key_col_indices = [self.column_names.index(col) for col in key_column_names]
        other_key_col_indices = [other_table.column_names.index(col) for col in key_column_names]

        # find indices of non-key columns in each table.
        non_key_col_indices = []
        non_other_key_col_indices = []

        for col in self.column_names:
            col_idx = self.column_names.index(col)
            if col_idx not in key_col_indices:
                non_key_col_indices.append(col_idx)
        
        for col in other_table.column_names:
            col_idx = other_table.column_names.index(col)
            if col_idx not in other_key_col_indices:
                non_other_key_col_indices.append(col_idx)

        # get column names to combined table.
        self_col_names = self.column_names
        other_col_names = [other_table.column_names[col] for col in non_other_key_col_indices]
        combined_col_names = self_col_names + other_col_names
        
       # create new MyPyTable object to fill with joined tables.
        combined_table = MyPyTable(combined_col_names, []) 

        # begin full outer join of table values.        
        for row1 in self.data:
            combined_key1 = tuple(row1[key] for key in key_col_indices) # get values in all key cols in the self table.
            
            # default match is false.
            match = False
            
            for row2 in other_table.data:
                combined_key2 = tuple(row2[key] for key in other_key_col_indices) # get values in all key cols in the other table.

                # if the values in combined key for table 1 == values in combined key for table 2, merge the rows.
                if combined_key1 == combined_key2:
                    match = True
                    matched.add(combined_key2) # add the combined key to matched so we know we've seen it already. 
                    extra_row2_cols = [row2[key] for key in non_other_key_col_indices]
                    combined_table.data.append(row1 + extra_row2_cols) # we only need to get the values in non-key cols of other table because row1 and row2 share key col values.
                    
            # first, make sure to get all non-matched rows from self table onto the new combined table. aka, handle self table.
            if not match:
                extra_row2_cols = ["NA"] * len(non_other_key_col_indices)
                combined_table.data.append(row1 + extra_row2_cols)
                
        # now, make sure to get all non-matched rows from the other table onto the new combined table. aka, handle other table.
        for row in other_table.data:
            combined_key2 = tuple(row[key] for key in other_key_col_indices) # get values in all key cols in the other table.

            # if we haven't seen this key yet, add row.
            if combined_key2 not in matched:
                
                # (1) fill all slots for each column in the combined table with "NA".
                combined_row = ["NA"] * len(combined_table.column_names)

                # (2) fill all columns of the current row in combined table with its value.
                # this is for filling in all key columns.
                # recall other_key_col_indices is a list of indices of key columns for other table.
                # 'i' is the "index" (the position in the key col list), 'row_idx' is the "value" (the actual column index from the key col list)
                for i, row_idx in enumerate(other_key_col_indices):
                    key_col_idx = key_col_indices[i] 
                    val = row[row_idx] # grab the value in the correct column ('row_idx') of the current row.
                    combined_row[key_col_idx] = val # assign value to the correct column in the combined table.

                # this is for unique columns in the other table.
                # append these columns after the columns in the self table
                for i, row_idx in enumerate(non_other_key_col_indices):
                    next_col_idx = len(self.column_names) + i
                    val = row[row_idx] # grab the value in the correct column ('row_idx') of the current row.
                    combined_row[next_col_idx] = val

                combined_table.data.append(combined_row)

        return combined_table


    def get_col_with_missing_values(self):
        """
        Purpose:
            Find all columns with missing values in them.

        Parameters:
            NA

        Returns:
            col_idx_with_missing_vals (list): a list of indices corresponding to columns with missing values.
            col_name_with_missing_vals (list): a list of the names of columns (strings) with missing values. 
        """
        col_idx_with_missing_vals = []
        col_name_with_missing_vals = []

        for col_idx, col_val in enumerate(self.column_names):
            for row in self.data:
                if row[col_idx] == 'NA':
                    col_idx_with_missing_vals.append(col_idx)
                    if col_val not in col_name_with_missing_vals:
                        col_name_with_missing_vals.append(col_val)
            

        return col_idx_with_missing_vals, col_name_with_missing_vals
    

    def convert_to_categorical(self, column):
        '''
        Purpose:
            Convert the continuous values in a column into categorical values.

        Parameters: 
            column: the column for which we will convert its continuous values into categorical values.

        Returns: 
            the table, but with the given column converted to categorical values now.
        '''
        # find the index of the given column in the table.
        col_idx = self.column_names.index(column)

        # iterate through each row in the dataset.
        for row in self.data:
            
            # find the mpg val of the current row
            mpg_val = row[col_idx]

            # assign categorical categories (1-10) of each instance in the column based on its continuous mpg value. 
            if mpg_val <=13:
                row[col_idx] = 1
            elif 13 <= mpg_val <= 14:
                row[col_idx] = 2
            elif 14 < mpg_val <= 16:
                row[col_idx] = 3
            elif 16 < mpg_val <= 19:
                row[col_idx] = 4
            elif 19 < mpg_val <= 23:
                row[col_idx] = 5
            elif 23 < mpg_val <= 26:
                row[col_idx] = 6
            elif 26 < mpg_val <= 30:
                row[col_idx] = 7
            elif 30 < mpg_val <= 36:
                row[col_idx] = 8
            elif 36 < mpg_val <= 44:
                row[col_idx] = 9
            else:
                row[col_idx] = 10
                          
    
    def calculate_bin_frequency(self, column_name, bin_val):
        '''
        Purpose: 
            Compute the frequencies (aka, number of instances) in a given bin.

        Parameters:
            column_name (str): the name of the column that we want to split into a bin. 
            bin_val (int): the bin value.

        Returns:
            frequency (list): the number of instances in a given bin. 
        '''
        # find the index of the given column in the table.
        col_idx = self.column_names.index(column_name)

        # initialize the counter to 0.
        frequency = 0

        # iterate through all rows in the table.
        for row in self.data: 
            # get current row value in the given column.
            row_val = row[col_idx]
            
            # if the current row value in the given column is equal to the bin value, add 1 to the counter.            
            if row_val == bin_val:
                frequency += 1

        return frequency

    
    def new_calculate_bin_frequencies(self, num_bins, values, cutoff_pts):
        '''
        Purpose:
            Calculate frequencies (aka, the number of instances) for all bins in the dataset.
        
        Parameters: 
            num_bins (int): the number of bins / categories to split the data into.
            values (list): the values that we want to put into the bins. 
            cutoff_pts (list): the ranges of values for each bin.

        Returns:
            frequencies (list): the frequencies (aka, the number of instances) in each bin. 
        '''
        # initialize a list of size 'num_bins' with 0s.
        frequencies = [0] * num_bins

        # iterate through all values.
        for val in values:
            
            # iterate through all bins.
            for bin in range(num_bins):

                # case 1: handle values that fall into any bin except for the last bin.
                if bin < num_bins - 1 and cutoff_pts[bin] <= val < cutoff_pts[bin + 1]:
                    frequencies[bin] += 1
                    break

                # case 2: handle values that fall into the last bin.
                elif bin == num_bins - 1 and cutoff_pts[bin] <= val <= cutoff_pts[bin + 1]:
                    frequencies[bin] += 1
                    break

        return frequencies
        



