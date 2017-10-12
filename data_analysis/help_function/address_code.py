import xlrd
address_file = xlrd.open_workbook('/Users/chandler/Desktop/ipRegion.xlsx')
sheet0 = address_file.sheet_by_index(0)
address = set(sheet0.col_values(0)[1:])
address_dict = {}
count = 0
for i in address:
    count += 1
    address_dict[count] = i

for k,v in address_dict.items():
    print(k,v)
