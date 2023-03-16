import pandas as pd
import numpy as np
from IPython.display import HTML, display

def print_confusion(cm):
    column_names = pd.DataFrame([['Predicted', '-'],
                                 ['Predicted', '+']],
                                 columns=['', ''])
    row_names = pd.DataFrame([['Observed', '-'],
                              ['Observed', '+']],
                              columns=['', ''])
    columns = pd.MultiIndex.from_frame(column_names)
    index = pd.MultiIndex.from_frame(row_names)
    display(pd.DataFrame(cm, index=index, columns=columns))

def get_score(cm):
    return (cm[0,0]+cm[1,1])/np.sum(cm)

def get_fp_rate(cm):
    return cm[0,1]/(cm[0,0]+cm[0,1])

def get_fn_rate(cm):
    return cm[1,0]/(cm[1,1]+cm[1,0])

def get_f1_score(cm):
    TP = cm[1,1]
    return TP/(TP+.5*(cm[1,0]+cm[0,1]))

def misclassification(cm):
    return (cm[0,1]+cm[1,0])/(cm[1,0]+cm[0,1]+cm[0,0]+cm[1,1])

def display_model_stats(cm, name):
    '''`cm` is a confusion matrix'''
    display(HTML('''
        <div class="row">
        <style scoped>
            tr th {{
                font-weight: 600;
                text-align: center;
            }}
            thead th {{
                font-weight: bold;
                text-align: center;
            }}
            .rotate {{
                transform: rotate(-180deg);
                writing-mode: vertical-rl;
                margin: 0em;
            }}
            .row {{ display: flex; }}
            .column {{ padding: 5px; }}
        </style>
        <div class="column">
            <table>
            <thead>
                <th colspan="4">{0}</th>
            </thead>
            <tr>
                <td rowspan="2" colspan="2"></td>
                <th colspan="2">Predicted</th>
            </tr>
            <tr>
                <th>-</td>
                <th>+</td>
            </tr>
            <tr>
                <th rowspan="2"><p class="rotate">Observed</p></th>
                <th>-</td>
                <td>{1}</td>
                <td>{2}</td>
            </tr>
            <tr>
                <th>+</td>
                <td>{3}</td>
                <td>{4}</td>
            </tr>
            </table>
        </div>
        <div class="column">
            <table>
            <thead>
                <th colspan="2">{0}</th>
            </thead>
            <tr>
                <th>Total accuracy</th>
                <td>{5:.3}</td>
            </tr>
            <tr>
                <th>False positive rate</th>
                <td>{6:.3}</td>
            </tr>
            <tr>
                <th>False negative rate</th>
                <td>{7:.3}</td>
            </tr>
            </table>
        </div>
        </div>
    '''.format(
        name,
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
        get_score(cm), get_fp_rate(cm), get_fn_rate(cm)
    )))