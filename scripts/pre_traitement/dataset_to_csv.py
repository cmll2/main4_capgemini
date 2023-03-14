import my_class
import sys

my_csv = 'extraction_csv/données.csv'

data_path = sys.argv[1]

my_datas = my_class.dataset(data_path)

my_csv = my_datas.analyze_dataset(my_csv)

sys.exit()