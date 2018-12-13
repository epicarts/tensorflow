#사용자 그룹 만들기
#같은 상품을 구매한 영국에서 거주하는 사용자들 그룹 추출
#InvoiceNo(거래 고유번호)	StockCode(상품코드)
#Description(상품명)	Quantity(판개 개수)
#InvoiceDate(거래 일시) UnitPrice(가격)	CustomerID(구매 ID)	Country(지역)

#1.StockCode를 이용해서 customerID 를 군집화하기
'''
user_product_dic : 딕셔너리 {사용자 ID : 구입한 상품코드}
product_user_dic : 딕셔너리 {상품코드 : 구입한 사용자 ID}
product_per_user_li : [사용자가 구매한 상품 개수]
product_id_name_dic : 딕셔너리 {상품코드 : 상품명}
'''

#기초 통계량은 scipy의 stats 모듈에서 제공하는 describe 함수로 쉽게 살펴볼 수 있음
scipy.stats.describe(a, axis=0, ddof=1, bias=True, nan_policy='propagate')

import time
from scipy import stats

user_product_dic = {}
product_user_dic = {}
product_per_user_li = []
product_id_name_dic ={}

for line in open('Online_Retail3.txt'):
    '''
    if ('q' == input("q 탈출 인풋")): #for문 탈출 .....
        break
    '''
    line_items = line.strip().split('\t')
    user_code = line_items[6]
    product_id = line_items[1]
    product_name = line_items[2]
    print(line_items)
    #사용자 id 가 없을 경우 무시
    if len(user_code) == 0:
        continue

    #
    country = line_items[7]
    if country != 'United Kingdom':
        continue


    #연도에 에러 파일 헤더 무시
    try:
        invoice_year = time.strptime(line_items[4], '%Y-%m-%d %H:%M').tm_year

    except ValueError:
        continue
    #2011년에 일어난 구매가 아닌건 무시
    if invoice_year != 2011:
        continue

    #읽은 정보로 데이터 구조를 채움
    #상품 가짓수를 고려하므로 상품 코드를 셋으로 가짐
    user_product_dic.setdefault(user_code, set())
    user_product_dic[user_code].add(product_id)

    product_user_dic.setdefault(product_id, set())
    product_user_dic[product_id].add(user_code)

    product_id_name_dic[product_id] = product_name



product_per_user_li = [len(x) for x in user_product_dic.values()]
product_per_user_li

print('# of user:', len(user_product_dic))
print('# of user:', len(product_user_dic))
print(stats.describe(product_per_user_li))


from collections import Counter
import matplotlib.pyplot as plt

plot_data_all = Counter(product_per_user_li)
plot_data_x = list(plot_data_all.keys())
plot_data_y = list(plot_data_all.values())
plt.xlabel('고유 상품 가짓수')
plt.ylabel('사용자수')
plt.scatter(plot_data_x, plot_data_y, marker='+')
plt.show()
