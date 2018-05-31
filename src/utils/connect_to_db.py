# import pyodbc
import pypyodbc as pyodbc

# Connection info
server = 'AZ01SV0160\GWSI'
db = 'DB name'
user = 'USERNAME'
pword = 'PASSWORD'
# connection object
conn = pyodbc.connect("DRIVER={{SQL Server}};SERVER={0}; database={1}; \
       trusted_connection=yes;UID={2};PWD={3}".format(server, db, user, pword))

# Cursor object. Can use for reading queries in Pandas	   
cursor = conn.cursor()


# Sample Query 
stmt = """select * from NBSlottingResponse
order by CreatedDate desc"""
# Excute Query here
df = pd.read_sql(stmt, conn)
df
