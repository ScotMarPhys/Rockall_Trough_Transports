def make_attr(reg_short,reg_long,end_points,data_used,CT_ref,SA_ref,rho_ref=1027.4):
    attrs_Q = {'name':f'Q_{reg_short}',
                   'long_name':f'{reg_short} Volume Transport',
                    'units':'Sv',
                    'Description':f'{reg_long} volume transport'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'
                  }
    attrs_Qh = {'name': f'Qh_{reg_short}',
                'long_name': f'{reg_short} Heat Flux',
                'units':'PW',
                'Description':f'Heat flux {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'
                   }

    attrs_Qf = {'name': f'Qf_{reg_short}',
                'long_name': f'{reg_short} Freshwater flux',
                'units':'PW',
                'Description':f'Freshwater flux {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'
                }

    attrs_QS = {'name': f'QS_{reg_short}',
                'long_name': f'{reg_short} salt flux',
                'units':'PW',
                'Description':f'salt flux {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'}

    attrs_q = {'name':f'q_{reg_short}',
                   'long_name':f'{reg_short} Volume Transport per cell',
                    'units':'Sv',
                    'Description':f'Volume transport per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                  }
    attrs_qh = {'name': f'qh_{reg_short}',
                'long_name': f'{reg_short} Heat Flux per cell',
                'units':'PW',
                'Description':f'Heat flux per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                f' Reference temperature {CT_ref}degC',}


    attrs_qf = {'name': f'qf_{reg_short}',
                'long_name': f'{reg_short} Freshwater flux per cell',
                'units':'Sv',
                'Description':f'Freshwater flux per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                f' Reference absolute salinity {SA_ref} (g/kg)',
                }

    attrs_qS = {'name': f'qS_{reg_short}',
                'long_name': f'{reg_short} salt flux per cell',
                'units':'Sv',
                'Description':f'salt flux per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                f' Reference density {rho_ref} (kg/m^3)',}
    return attrs_Q,attrs_Qh,attrs_Qf,attrs_QS,attrs_q,attrs_qh,attrs_qf,attrs_qS