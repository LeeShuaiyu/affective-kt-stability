import numpy as np
import math


class DATA(object):
    def __init__(self, seqlen, separate_char):
        self.separate_char = separate_char
        self.seqlen = seqlen

    '''
    data format:
    length
    KC sequence
    answer sequence
    exercise sequence
    it sequence
    at sequence
    '''

    def load_data(self, path):
        a_data = []
        s_data = []
        e_data = []
        it_data = []
        at_data = []

        fru_data=[]
        conf_data=[]
        conc_data=[]
        bor_data=[]

        sd_data =[]
        qd_data =[]

        tp_data=[]

        stu_data=[]
        pre_data=[]
        att_data=[]

        with open(path, 'r') as f_data:
            for lineID, line in enumerate(f_data):
                line = line.strip()
                if lineID % 16 != 0:
                    line_data = line.split(self.separate_char)
                    if len(line_data[len(line_data) - 1]) == 0:
                        line_data = line_data[:-1]

                if lineID % 16 == 2:
                    A = line_data
                elif lineID % 16 ==1:
                    S =line_data
                elif lineID % 16 == 3:
                    E = line_data
                elif lineID % 16 == 4:
                    IT = line_data
                elif lineID % 16 == 5:
                    AT = line_data
                elif lineID % 16 ==6:
                    BOR =line_data
                elif lineID % 16 == 7:
                    CONC = line_data
                elif lineID % 16 == 8:
                    CONF = line_data
                elif lineID % 16 == 9:
                    FRU = line_data
                elif lineID % 16 == 10:
                    QD = line_data
                elif lineID % 16 == 11:
                    SD = line_data
                elif lineID % 16 == 12:
                    TP = line_data
                elif lineID % 16 == 13:
                    STU = line_data
                elif lineID % 16 == 14:
                    PRE = line_data
                elif lineID % 16 == 15:
                    ATT = line_data

                    # start split the data
                    n_split = 1
                    total_len = len(A)
                    if total_len > self.seqlen:
                        n_split = math.floor(len(A) / self.seqlen)
                        if total_len % self.seqlen:
                            n_split = n_split + 1

                    for k in range(n_split):
                        answer_sequence = []
                        exercise_sequence = []
                        skill_sequence =[]
                        it_sequence = []
                        at_sequence = []

                        fru_sequence = []
                        conf_sequence = []
                        conc_sequence = []
                        bor_sequence = []

                        sd_sequence = []
                        qd_sequence = []
                        tp_sequence = []

                        stu_sequence = []
                        pre_sequence = []
                        att_sequence = []


                        if k == n_split - 1:
                            end_index = total_len
                        else:
                            end_index = (k + 1) * self.seqlen
                        # choose the sequence length is larger than 2
                        if end_index - k * self.seqlen > 2:
                            for i in range(k * self.seqlen, end_index):
                                answer_sequence.append(int(A[i]))
                                exercise_sequence.append(int(E[i]))
                                skill_sequence.append(int(S[i]))
                                it_sequence.append(int(IT[i]))
                                at_sequence.append(int(AT[i]))

                                bor_sequence.append(float(BOR[i]))
                                conc_sequence.append(float(CONC[i]))
                                conf_sequence.append(float(CONF[i]))
                                fru_sequence.append(float(FRU[i]))
                                sd_sequence.append(int(SD[i]))
                                qd_sequence.append(int(QD[i]))
                                tp_sequence.append(int(TP[i]))
                                stu_sequence.append(int(STU[i]))

                                pre_sequence.append(int(PRE[i]))
                                att_sequence.append(int(ATT[i]))


                            a_data.append(answer_sequence)
                            e_data.append(exercise_sequence)
                            s_data.append(skill_sequence)
                            it_data.append(it_sequence)
                            at_data.append(at_sequence)
                            bor_data.append(bor_sequence)
                            conc_data.append(conc_sequence)
                            conf_data.append(conf_sequence)
                            fru_data.append(fru_sequence)
                            qd_data.append(qd_sequence)  
                            sd_data.append(sd_sequence)
                            tp_data.append(tp_sequence)   
                            stu_data.append(stu_sequence)
                            pre_data.append(pre_sequence)
                            att_data.append(att_sequence)

        def pad_sequences(sequence_list):
            padded = np.zeros((len(sequence_list), self.seqlen))
            for j, dat in enumerate(sequence_list):
                padded[j, :len(dat)] = dat
            return padded

        # data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        a_dataArray = pad_sequences(a_data)
        e_dataArray = pad_sequences(e_data)
        it_dataArray = pad_sequences(it_data)
        s_dataArray = pad_sequences(s_data)
        at_dataArray = pad_sequences(at_data)

        fru_dataArray = pad_sequences(fru_data)
        conf_dataArray = pad_sequences(conf_data)
        conc_dataArray = pad_sequences(conc_data)
        bor_dataArray = pad_sequences(bor_data)

        sd_dataArray = pad_sequences(sd_data)
        qd_dataArray = pad_sequences(qd_data)
        tp_dataArray = pad_sequences(tp_data)

        stu_dataArray = pad_sequences(stu_data)
        pre_dataArray = pad_sequences(pre_data)
        att_dataArray = pad_sequences(att_data)

        return a_dataArray, e_dataArray,s_dataArray ,it_dataArray, at_dataArray,bor_dataArray,conc_dataArray,conf_dataArray,fru_dataArray,qd_dataArray,sd_dataArray,tp_dataArray,stu_dataArray,pre_dataArray,att_dataArray
