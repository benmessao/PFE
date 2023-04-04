import pandas as pd
import numpy as np


def new_features(merged):
    
    print(merged.shape)
    
    sender = []
    max_speed1 = []
    max_speed2 = []
    nb_packets_sent = []


    for i in range(len(merged['sender'].unique())):
        print(i)
        sender.append(merged['sender'].unique()[i])
        max_speed1.append(merged[merged['sender'] == merged['sender'].unique()[i]]['spd_x_send'].max())
        max_speed2.append(merged[merged['sender'] == merged['sender'].unique()[i]]['spd_y_send'].max())
        nb_packets_sent.append(len(merged[merged['sender'] == merged['sender'].unique()[i]]))

    senders = pd.DataFrame()
    senders['sender'] = sender
    senders['max_speed1'] = max_speed1
    senders['max_speed2'] = max_speed2
    senders['nb_packets_sent'] = nb_packets_sent
    
    merged = merged.merge(senders, on='sender')
    
    merged['frequency1'] = abs(merged['spd_x_send'] - merged['max_speed1']/2)
    merged['frequency2'] = abs(merged['spd_y_send'] - merged['max_speed2']/2)
    
    merged = merged.sort_values(by=['sender', 'receiver', 'sendTime'])
    merged['time_diff'] = merged.groupby(['sender', 'receiver'])['sendTime'].diff()
    
    merged['distRealSR1'] = merged['pos_x_rec'] - merged['pos_x_send']
    merged['distRealSR2'] = merged['pos_y_rec'] - merged['pos_y_send']
    merged['diffSpdSR1'] = merged['spd_x_rec'] - merged['spd_x_send']
    merged['diffSpdSR2'] = merged['spd_y_rec'] - merged['spd_y_send']
    merged['diffAclSR1'] = merged['acl_x_rec'] - merged['acl_x_send']
    merged['diffAclSR2'] = merged['acl_y_rec'] - merged['acl_y_send']
    merged['diffHedSR1'] = merged['hed_x_rec'] - merged['hed_x_send']
    merged['diffHedSR2'] = merged['hed_y_rec'] - merged['hed_y_send']

    merged['deltaPosRec1'] = abs(merged['pos_x_rec'] - merged['pos_x_rec_f'])
    merged['deltaPosRec2'] = abs(merged['pos_y_rec'] - merged['pos_y_rec_f'])
    merged['deltaSpdRec1'] = abs(merged['spd_x_rec'] - merged['spd_x_rec_f'])
    merged['deltaSpdRec2'] = abs(merged['spd_y_rec'] - merged['spd_y_rec_f'])
    merged['deltaAclRec1'] = abs(merged['acl_x_rec'] - merged['acl_x_rec_f'])
    merged['deltaAclRec2'] = abs(merged['acl_y_rec'] - merged['acl_y_rec_f'])
    merged['deltaHedRec1'] = abs(merged['hed_x_rec'] - merged['hed_x_rec_f'])
    merged['deltaHedRec2'] = abs(merged['hed_y_rec'] - merged['hed_y_rec_f'])

    merged['deltaPos1'] = abs(merged['pos_x_send'] - merged['pos_x_send_f'])
    merged['deltaPos2'] = abs(merged['pos_y_send'] - merged['pos_y_send_f'])
    merged['deltaSpd1'] = abs(merged['spd_x_send'] - merged['spd_x_send_f'])
    merged['deltaSpd2'] = abs(merged['spd_y_send'] - merged['spd_y_send_f'])
    merged['deltaAcl1'] = abs(merged['acl_x_send'] - merged['acl_x_send_f'])
    merged['deltaAcl2'] = abs(merged['acl_y_send'] - merged['acl_y_send_f'])
    merged['deltaHed1'] = abs(merged['hed_x_send'] - merged['hed_x_send_f'])
    merged['deltaHed2'] = abs(merged['hed_y_send'] - merged['hed_y_send_f'])

    merged['distance'] = np.sqrt(merged['distRealSR1']**2 + merged['distRealSR2']**2)
    merged['difSpeed'] = np.sqrt(merged['diffSpdSR1']**2 + merged['diffSpdSR2']**2)
    merged['estAoA'] = np.arctan(merged['distRealSR2']/merged['distRealSR1'])
    
    print(merged.shape)

    return(merged)