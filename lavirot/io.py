import pandas as pd
import numpy as np


__all__ = ['load_from_xltrc']


def load_from_xltrc(n, file, sheet_name='XLUseKCM'):
    """Load bearing from xltrc.

    This method will construct a bearing loading the coefficients
    from an xltrc file.

    Parameters
    ----------
    n: int
        Node in which the bearing will be inserted.
    file : str
        File path name.
    sheet_name : str
        Bearing sheet name. Default is 'XLUseKCM'.

    Returns
    -------
    bearing : lr.BearingElement
        A bearing element.

    Examples
    --------
    """
    # TODO Check .xls units to see if argument provided is consistent

    df = pd.read_excel(file, sheet_name=sheet_name)

    start_col = 0

    if sheet_name == 'XLTFPBrg':
        start_col = 2

    # locate were table starts
    for i, row in df.iterrows():
        if row.iloc[start_col] == 'Speed':
            start = i + 2

    df_bearing = pd.DataFrame(df.iloc[start:, start_col:])
    if sheet_name == 'XLTFPBrg':
        df_bearing = df_bearing.iloc[:10]
    df_bearing = df_bearing.rename(columns=df.loc[start - 2])
    df_bearing = df_bearing.dropna(axis=0, thresh=5)
    df_bearing = df_bearing.dropna(axis=1, thresh=5)

    if df.iloc[start - 1, 1] == 'lb/in':
        for col in df_bearing.columns:
            if col != 'Speed':
                df_bearing[col] = df_bearing[col] * 175.126835

    df_bearing['Speed'] = df_bearing['Speed'] * 2 * np.pi / 60

    w = df_bearing.Speed.values
    kxx = df_bearing.Kxx.values
    kxy = df_bearing.Kxy.values
    kyx = df_bearing.Kyx.values
    kyy = df_bearing.Kyy.values
    cxx = df_bearing.Cxx.values
    cxy = df_bearing.Cxy.values
    cyx = df_bearing.Cyx.values
    cyy = df_bearing.Cyy.values

    return dict(n=n, w=w, kxx=kxx, kxy=kxy, kyx=kyx, kyy=kyy,
                cxx=cxx, cxy=cxy, cyx=cyx, cyy=cyy)