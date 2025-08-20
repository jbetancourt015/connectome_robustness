"""
    This script saves the FlyWire brain regions as a dictionary.
-------------------------------------------------------------------------------
created on:
    Mon 30 Jun 2025
-------------------------------------------------------------------------------
last change:
    Mon 30 Jun 2025
-------------------------------------------------------------------------------
notes:
-------------------------------------------------------------------------------
contributors:
    Jose:
        name:       Jose Betancourt
        email:      jose.betancourtvalencia@yale.edu
-------------------------------------------------------------------------------
"""
import pickle

region_map = {
    # Optic Lobe
    **dict.fromkeys(
        ['AME_L','LA_L','LO_L','LOP_L','ME_L',
         'AME_R','LA_R','LO_R','LOP_R','ME_R'],
        'Optic Lobe'
    ),

    # Lateral Complex
    **dict.fromkeys(
        ['BU_L','LAL_L','GA_L',
         'BU_R','LAL_R','GA_R'],
        'Lateral Complex'
    ),

    # Lateral Horn
    **dict.fromkeys(['LH_L','LH_R'], 'Lateral Horn'),

    # Periesophageal Neuropils (left, right, + saddle/prow)
    **dict.fromkeys(
        ['CAN_L','AMMC_L','FLA_L',
         'CAN_R','AMMC_R','FLA_R',
         'SAD','PRW'],
        'Periesophageal Neuropils'
    ),

    # Inferior Neuropils
    **dict.fromkeys(
        ['ICL_L','IB_L','ATL_L','CRE_L','SCL_L',
         'ICL_R','IB_R','ATL_R','CRE_R','SCL_R'],
        'Inferior Neuropils'
    ),

    # Ventromedial Neuropils
    **dict.fromkeys(
        ['VES_L','GOR_L','SPS_L','IPS_L','EPA_L',
         'VES_R','GOR_R','SPS_R','IPS_R','EPA_R'],
        'Ventromedial Neuropils'
    ),

    # Mushroom Body
    **dict.fromkeys(
        ['MB_PED_L','MB_VL_L','MB_ML_L','MB_CA_L',
         'MB_PED_R','MB_VL_R','MB_ML_R','MB_CA_R'],
        'Mushroom Body'
    ),

    # Antennal Lobe
    **dict.fromkeys(['AL_L','AL_R'], 'Antennal Lobe'),

    # Superior Neuropils
    **dict.fromkeys(
        ['SLP_L','SIP_L','SMP_L',
         'SLP_R','SIP_R','SMP_R'],
        'Superior Neuropils'
    ),

    # Ventrolateral Neuropils
    **dict.fromkeys(
        ['AVLP_L','PVLP_L','WED_L','PLP_L','AOTU_L',
         'AVLP_R','PVLP_R','WED_R','PLP_R','AOTU_R'],
        'Ventrolateral Neuropils'
    ),

    # Central Complex
    **dict.fromkeys(['NO','PB','EB','FB'], 'Central Complex'),

    # Gnathal Ganglia
    'GNG': 'Gnathal Ganglia',

    # Ocelli
    'OCG': 'Ocelli',

    # Other Regions (unassigned)
    'UNASGD': 'Other Regions',
}

with open('../processed_data/region_map.pkl', 'wb') as f:
    pickle.dump(region_map, f)