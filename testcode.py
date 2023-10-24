if (idx1) % int(self.batch_num / 5) == 0:
    self.type_3_sampled_for_balance = self.type_3_data[np.random.choice(len(self.type_3_data), int((self.type_1_data_len  self.type_2_data_len)*1.5),replace=False)]
    self.type_3_batch_indexes = GetBatchIndexes(self.type_3_data_len, self.batch_num)
