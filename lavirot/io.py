import pandas as pd
import numpy as np
from .materials import Material

__all__ = ['load_bearing_seals_from_xltrc', 'load_disks_from_xltrc',
           'load_shaft_from_xltrc']


def table_limits_xltrc(df, start_col=0):
    """Find where table starts for xl files."""
    for i, row in df.iterrows():
        if row.iloc[start_col] == 'Speed':
            start = i
    for i in range(start, 100):
        if pd.isna(df.iloc[i, start_col]):
            end = i - 1
            break

    return start, end


def load_bearing_seals_from_xltrc(file, sheet_name='XLUseKCM'):
    """Load bearing from xltrc.

    This method will construct a bearing loading the coefficients
    from an xltrc file.

    Parameters
    ----------
    n: int
        Node in which the bearing will be inserted.
    file : str
        File path name.
    sheet_name : str Bearing sheet name. Default is 'XLUseKCM'.
    Returns
    -------
    bearing : lr.BearingElement
        A bearing element.

    Examples
    --------
    """
    # TODO Check .xls units to see if argument provided is consistent

    if sheet_name == 'XLLaby':
        return load_from_excel_laby(file)

    df = pd.read_excel(file, sheet_name=sheet_name)

    col_start = 0

    if sheet_name == 'XLTFPBrg':
        col_start = 2

    table_start, table_end = table_limits_xltrc(df, col_start)
    table_start = table_start + 2

    df_bearing = pd.DataFrame(df.iloc[table_start:, col_start:])
    if sheet_name == 'XLTFPBrg':
        df_bearing = df_bearing.iloc[:10]
    df_bearing = df_bearing.rename(columns=df.loc[table_start - 2])
    df_bearing = df_bearing.dropna(axis=0, thresh=5)
    df_bearing = df_bearing.dropna(axis=1, thresh=5)

    if df.iloc[table_start - 1, 1] == 'lb/in':
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

    return dict(w=w, kxx=kxx, kxy=kxy, kyx=kyx, kyy=kyy,
                cxx=cxx, cxy=cxy, cyx=cyx, cyy=cyy)


def load_from_excel_laby(file):
    df = pd.read_excel(file)
    table_start, table_end = table_limits_xltrc(df)

    df_seal = pd.DataFrame(df.iloc[table_start + 2:table_end, :9])
    df_seal = df_seal.rename(columns=df.iloc[table_start, :9])

    kwargs = {}
    for k, v in df_seal.to_dict().items():
        if k == 'Speed':
            kwargs['w'] = (2 * np.pi / 60) * \
                          np.fromiter(v.values(), dtype=np.float)
        else:
            kwargs[k.lower()] = np.fromiter(v.values(), dtype=np.float)

    return kwargs


def load_disks_from_xltrc(file, sheet_name='More'):
    """Load lumped masses from xltrc.

    This method will construct a disks list with loaded data
    from a xltrc file.

    Parameters
    ----------
    file : str
        File path name.
    sheet_name : str
        Masses sheet name. Default is 'More'.

    Returns
    -------
    disks : list
        List with the shaft elements.

    Examples
    --------
    """
    df = pd.read_excel(file, sheet_name=sheet_name)

    df_masses = pd.DataFrame(df.iloc[4:, :4])
    df_masses = df_masses.rename(columns=df.iloc[1, 1:4])
    df_masses = df_masses.rename(columns={' Added Mass & Inertia': 'n'})

    # convert to SI units
    if df.iloc[2, 1] == 'lbm':
        df_masses['Mass'] = df_masses['Mass'] * 0.45359237
        df_masses['Ip'] = df_masses['Ip'] * 0.00029263965342920005
        df_masses['It'] = df_masses['It'] * 0.00029263965342920005

    return df_masses


def load_shaft_from_xltrc(file, sheet_name='Model'):
    """Load shaft from xltrc.

    This method will construct a shaft loading the geometry
    and materials from a xltrc file.

    Parameters
    ----------
    file : str
        File path name.
    sheet_name : str
        Shaft sheet name. Default is 'Model'.

    Returns
    -------
    shaft : list
        List with the shaft elements.

    Examples
    --------
    """
    df = pd.read_excel(file, sheet_name=sheet_name)

    geometry = pd.DataFrame(df.iloc[19:])
    geometry = geometry.rename(columns=df.loc[18])
    geometry = geometry.dropna(thresh=3)

    material = df.iloc[3:13, 9:15]
    material = material.rename(columns=df.iloc[0])
    material = material.dropna(axis=0, how='all')

    # change to SI units
    if df.iloc[1, 1] == 'inches':
        for dim in ['length', 'od_Left', 'id_Left',
                    'od_Right', 'id_Right']:
            geometry[dim] = geometry[dim] * 0.0254

        geometry['axial'] = geometry['axial'] * 4.44822161

        for prop in ['Elastic Modulus E', 'Shear Modulus G']:
            material[prop] = material[prop] * 6894.757

        material['Density   r'] = material['Density   r'] * 27679.904

    colors = ['#636363', '#969696', '#bdbdbd', '#d9d9d9',
              '#636363', '#969696', '#bdbdbd', '#d9d9d9']
    materials = {}
    for i, mat in material.iterrows():
        materials[mat.Material] = Material(
            name=f'Material{mat["Material"]}',
            rho=mat['Density   r'],
            E=mat['Elastic Modulus E'],
            G_s=mat['Shear Modulus G'],
            color=colors[mat['Material']]
        )

    return (geometry, materials)
    shaft = [ShaftElement(
        el.length, el.id_Left, el.od_Left,
        materials[el.matnum], n=el.elemnum-1)
        for i, el in geometry.iterrows()]

    return shaft