import polars as pl


class SortedDS(pl.LazyFrame):
    def __init__(self, df, id='id'):
        super().__init__(df)
        self.id = id
        self.bp = df.with_row_indices()\
                    .filter(pl.col(id).ne(pl.col(id).shift(1)))\
                    .select(id, 'index')\
                    .collect()
    
    def slice_fl(self, first, last):
        return self.slice(first, last - first)
    
    def __len__(self):
        return self.select(pl.len()).collect().item()

    def chunks(self, size, with_name=False):
        for i in range(0, self.bp.shape[0], size):
            ret = self.slice_fl(self.bp['index'][i * size],
                                self.bp['index'][(i + 1) * size] if (i + 1) * size < len(self.bp) else len(self))
            if with_name:
                ret = (ret, self.bp['id'][i * size])
            yield ret