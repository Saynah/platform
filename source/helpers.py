import pandas as pd
import numpy as np

COLS_DX = ['highBPDiagnosed',
           'diabetesDiagnosed',
           'chdDiagnosed',
           'miDiagnosed',
           'anginaDiagnosed',
           'strokeDiagnosed',
           'emphysemaDiagnosed',
           'asthmaDiagnosed',
           'otherHDDiagnosed',
           'heartFailureDiagnosed']


def load_base_data(raw=False, minors=False):
    """load and prep base data for patients

    - remap diagnoses:
        yes -> True
        no -> False
        ['Inapplicable', 'DK', 'Refused', 'not ascertained'] -> NaN
    - exclude minors by default
    """
    df_base = pd.read_csv('data/meps_base_data.csv', index_col=0)

    if raw:
        return df_base

    df_dx = (df_base
             .loc[:, COLS_DX]
             .replace(['Inapplicable', 'DK', 'Refused', 'not ascertained'], np.nan)
             .replace('Yes', True)
             .replace('No', False)
             )
    df_base.loc[:, COLS_DX] = df_dx

    if not minors:
        df_base = df_base[df_base['age'] >= 18]

    df_base.set_index('id', inplace=True)

    assert set(df_base[COLS_DX].values.ravel().tolist()) == set([True, False, np.nan])
    assert df_base.index.is_unique

    return df_base


def load_meds_data(raw=False):
    """load and prep meds data

    - disambiguate drug names
    - collapse duplicate prescriptions per patient
    """
    _df_meds = pd.read_csv('data/meps_meds.csv', index_col=0)

    if raw:
        return _df_meds

    rxNickname = (_df_meds['rxName']
                  .str.upper()
                  .str.extract('([A-Z]{4,})', expand=False)
                  .str.strip()
                  )
    rxNickname.name = 'rxNickname'

    f = lambda df: pd.Series({
        'numPrescriptions': len(df),
        'originalNDCs': df['rxNDC'].unique()
    })

    df_meds = _df_meds.groupby(['id', rxNickname]).apply(f)

    assert df_meds.index.is_unique
    
    return df_meds