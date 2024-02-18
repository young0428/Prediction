#%%

# %%

class aaa:
    keys = "member"
    def __init__(self):
        self.keys = "member"
    def keys(self):
        return "method"
    
test = aaa()
print(test.keys)
print(test.keys())
    
    
    
    


# %%
