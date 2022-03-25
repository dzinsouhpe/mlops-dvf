import pandas as pd
import pickle
import json, sys

dvf_model = pickle.load(open('/bd-fs-mnt/project_repo/models/dvf/1/model.pkl', "rb"))

cli_input = json.loads('{"nb_pieces": ' + sys.argv[1] + ',"surface": ' + sys.argv[2] + ',"code_postal_31100": ' + sys.argv[3] + ',"code_postal_31200": ' + sys.argv[4] + ',"code_postal_31300": ' + sys.argv[5] + ',"code_postal_31400": ' + sys.argv[6] + ',"code_postal_31500": ' + sys.argv[7] + '}')

x = pd.DataFrame(cli_input, index=[0])
y =  dvf_model.predict(x)
print(int(y))