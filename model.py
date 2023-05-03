from torch import nn

class PostLN(nn.Module):
    def __init__(self, f, embed_dim):
        super().__init__()
        self.f = f
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, input):
        return self.ln(self.f(input) + input)

class PreLN(nn.Module):
    def __init__(self, f, embed_dim):
        super().__init__()
        self.f = f
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, input):
        return input + self.f(self.ln(input))

class ResiDual(nn.Module):
    def __init__(self, f, embed_dim):
        super().__init__()
        self.f = f
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x_ln, x_d):
        x_f = self.f(x_ln)
        x_d = x_d + x_f
        x_ln = self.ln(x_ln + x_f)
        return x_ln, x_d
