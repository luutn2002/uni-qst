import torch
from torch import nn
import numpy as np
import scipy as scp

class Basis(nn.Module):
    def __init__(self):
        super(Basis, self).__init__()
        '''
        Create a simple fock state basis.
        '''
    def forward(self, dimensions, n=None):
        if n is None: real = torch.ones(dimensions)
        else: 
            real = torch.zeros(dimensions)
            real[n] += 1
        imag = torch.zeros(dimensions)
        return torch.complex(real, imag)
    
class Destroy(nn.Module):
    def __init__(self):
        super(Destroy, self).__init__()
        '''
        Create a destroy operator.
        '''
    def forward(self, N, offset=0):
        data = torch.sqrt(torch.arange(offset+1, N+offset).type(torch.complex128))
        ind = torch.arange(1,N, dtype=torch.int32)
        ptr = torch.arange(N+1, dtype=torch.int32)
        ptr[-1] = N-1
        return torch.sparse_csr_tensor(ptr, ind, data, size=(N, N)).to_dense()
    
class Coherent(nn.Module):
    def __init__(self):
        super(Coherent, self).__init__()
        '''
        Create a coherent ket.
        '''
    def forward(self, N, alpha):
        x = Basis()(N, 0)
        a = Destroy()(N)
        alpha = torch.tensor([alpha], dtype=torch.complex128)
        D = torch.linalg.matrix_exp(alpha*a.adjoint() - torch.conj(alpha)*a)
        return torch.mul(D, x)[:,0]
    
class Thermal(nn.Module):
    def __init__(self):
        super(Thermal, self).__init__()
        '''
        Create a thermal.
        '''
    def forward(self, hilbert_size, mean_photon_number=None):
        if mean_photon_number == 0:
            return Fock()(hilbert_size, 0)
        else:
            if mean_photon_number is None:
                mean_photon_number = hilbert_size/2*torch.rand(1).item()
                
            i = torch.arange(hilbert_size)
            beta = torch.log(torch.tensor([1.0/mean_photon_number+1.0]))
            diags = torch.exp(-beta * i)
            diags = diags / torch.sum(diags, -1)
            # populates diagonal terms using truncated operator expression
            return torch.diag_embed(diags)

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()
        """
        Generates a cat state. For a detailed discussion on the definition
        see `Albert, Victor V. et al. “Performance and Structure of Single-Mode Bosonic Codes.” Physical Review A 97.3 (2018) <https://arxiv.org/abs/1708.05010>`_
        and `Ahmed, Shahnawaz et al., “Classification and reconstruction of quantum states with neural networks.” Journal <https://arxiv.org/abs/1708.05010>`_
    
        Args:
        -----
        N (int): Hilbert size dimension.
        alpha (complex64): Complex number determining the amplitude.
        S (int): An integer >= 0 determining the number of coherent states used
                 to generate the cat superposition. S = {0, 1, 2, ...}.
                 corresponds to {2, 4, 6, ...} coherent state superpositions.
        mu (int): An integer 0/1 which generates the logical 0/1 encoding of 
                  a computational state for the cat state.
        Returns:
        -------
        cat (:class:`qutip.Qobj`): Cat state density matrix
        """
        self.pi = torch.acos(torch.zeros(1))*2
    def random_alpha(self, inner_radius=0, radius=1):
        """
        Generates a random complex values within a circle
    
        Args:
            radius (float): Radius for the values
            inner_radius (float): Inner radius which defaults to 0.
        """
        radius = torch.FloatTensor(1).uniform_(inner_radius, radius)
        phi = torch.FloatTensor(1).uniform_(-self.pi, self.pi)
        
        return radius*torch.exp(1j * phi)
    
    def forward(self, hilbert_size, alpha=None, S=None, mu=None):
        if alpha == None:
            alpha = self.random_alpha(2, 3)

        if S == None:
            S = torch.randint(0, 3, (1,)).item()
      
        if mu is None:
            mu = torch.randint(0, 2, (1,)).item()

        kend = 2*S + 1
        cstates = 0*(Coherent()(hilbert_size, 0))

        for k in range(0, (kend + 1)//2):
            sign = 1

            if k >= S:
                sign = (-1) ** int(mu > 0.5)

            prefactor = torch.exp(1j * (self.pi / (S + 1)) * k)

            cstates += sign*Coherent()(hilbert_size, prefactor * alpha * (-((1j) ** mu)))
            cstates += sign*Coherent()(hilbert_size, -prefactor * alpha * (-((1j) ** mu)))

        rho = cstates.view(-1, 1) * cstates.view(-1, 1).adjoint()
        return rho*1/torch.trace(rho)

class Fock(nn.Module):
    def __init__(self):
        super(Fock, self).__init__()
         
    def forward(self, hilbert_size, n=None):
        if n == None:
            n = torch.randint(1, hilbert_size/2 + 1, (1,)).item()
        psi = Basis()(hilbert_size, n).view(-1, 1)

        return psi*psi.adjoint()
    
class CoherentDM(nn.Module):
    def __init__(self):
        super(CoherentDM, self).__init__()
         
    def forward(self, hilbert_size, alpha=None):
        if alpha == None:
            alpha = Cat.random_alpha(1e-6, 3)
        psi = Coherent()(hilbert_size, alpha).view(-1, 1)
        
        return psi*psi.adjoint()
    
class GKP(nn.Module):
    def __init__(self):
        super(GKP, self).__init__()
        self.pi = torch.acos(torch.zeros(1))*2
         
    def forward(self, hilbert_size, delta=None, mu = None):
        gkp = 0*Coherent()(hilbert_size, 0)

        c = torch.sqrt(self.pi/2)

        if mu is None:
            mu = torch.randint(2, (1,)).item()

        if delta is None:
            delta = ((0.2 - 0.5)*torch.rand(1)+0.5).item()

        zrange = torch.arange(-20, 20)
        
        #for loop took very long to process?
        for n1 in zrange:
            for n2 in zrange:        
                a = c*(2*n1 + mu + 1j*n2)
                alpha = Coherent()(hilbert_size, a)
                gkp += torch.exp(-delta**2*torch.abs(a)**2)*torch.exp(-1j*c**2 * 2*n1 * n2)*alpha
        
        gkp = gkp.view(-1, 1)
        rho = gkp*gkp.adjoint()
        return rho*1/torch.trace(rho)
    
    
class Binomial(nn.Module):
    def __init__(self):
        super(Binomial, self).__init__()
        
    def binom(self, n, k):
        n = torch.Tensor([n])
        k = torch.Tensor([k])
        return torch.exp(torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1))
                 
    def forward(self, hilbert_size, S=None, N=None, mu=None):
        if S == None:
            S = torch.randint(1, 10, (1,)).item()
    
        if N == None:
            Nmax = int((hilbert_size)/(S+1)) - 1
            try:
                N = torch.randint(2, Nmax, (1,)).item()
            except:
                N = Nmax

        if mu is None:
            mu = torch.randint(2, (1,)).item()

        c = 1/torch.sqrt(torch.tensor(2**(N+1)))

        psi = 0*Basis()(hilbert_size, 0)

        for m in range(N):
            psi += c*((-1)**(mu*m))*torch.sqrt(self.binom(N+1, m))*Basis()(hilbert_size, (S+1)*m)
        
        psi = psi.view(-1, 1)
        rho = psi*psi.adjoint()
        return rho*1/torch.trace(rho)
    
class Num(nn.Module):
    def __init__(self):
        super(Num, self).__init__()
        self.pi = torch.acos(torch.zeros(1))*2
        
    def get_random_num_prob(self, idx=None):
        states17 = torch.tensor([[(np.sqrt(7 - np.sqrt(17)))/np.sqrt(6), 0, 0, (np.sqrt(np.sqrt(17) - 1)/np.sqrt(6)), 0],
                    [0, (np.sqrt(9 - np.sqrt(17))/np.sqrt(6)), 0, 0, (np.sqrt(np.sqrt(17) - 3)/np.sqrt(6))]])


        statesM = torch.tensor([[0.5458351325482939, -3.7726009161224436e-9, 4.849511177634774e-8, \
        -0.7114411727633639, -7.48481181758003e-8, -1.3146003192319789e-8, \
        0.44172510726665587, 1.1545802803733896e-8, 1.0609402576342428e-8, \
        -0.028182506843720707, -6.0233214626778965e-9, -6.392041552216322e-9, \
        0.00037641909140801935, -6.9186916801058116e-9], \
        [2.48926815257019e-9, -0.7446851186077535, -8.040831059521339e-9, \
        6.01942995399906e-8, -0.5706020908811399, -3.151900508005823e-8, \
        -7.384935824733578e-10, -0.3460030551087218, -8.485651303145757e-9, \
        -1.2114327561832047e-8, 0.011798401879159238, -4.660460771433317e-9, \
        -5.090374160706911e-9, -0.00010758601713550998]])


        statesP = torch.tensor([[0., 0.7562859301326029, 0., 0., -0.5151947804474741, \
        -0.20807866860791188, 0.12704803323656158, 0.05101928893751686, \
        0.3171198939841734], [-0.5583217426728544, -0.0020589109231194413, \
        0., -0.7014041964402703, -0.05583041652626998, 0.0005664728465725445, \
        -0.2755044401850055, -0.3333309025086189, 0.0785824556163142]])

        statesP2 = torch.tensor([[-0.5046617350158988, 0.08380989527942606, -0.225295417417812, 0., \
        -0.45359477373452817, -0.5236866813756252, 0.2523308675079494, 0., \
        0.09562538828178244, 0.2172849136874009, 0., 0., 0., \
        -0.2793663175980869, -0.08280858231312467, -0.05106696128137072], \
        [-0.0014249418817930378, 0.5018692341095683, 0.4839749920101922, \
        -0.3874886488913531, 0.055390715144453026, -0.25780190053922486, \
        -0.08970154713375252, -0.1892386424818236, 0.10840637100094529, \
        -0.19963901508324772, -0.41852779130900664, -0.05747247660559087, 0., \
        -0.0007888071131354318, -0.1424131123943283, -0.0001441905475623907]])


        statesM2 = torch.tensor([[-0.45717455741713664, \
        np.complex(-1.0856965103853774e-6,1.3239037829080093e-6), \
        np.complex(-0.35772784377291084,-0.048007740168066144), \
        np.complex(-3.5459165445315755e-6,0.000012571453643232864), \
        np.complex(-0.5383420820794502,-0.24179040513272307), \
        np.complex(9.675641330014822e-7,4.569566899500361e-6), \
        np.complex(0.2587482691377581,0.313044506480362), \
        np.complex(4.1979351791851435e-6,-1.122460690803522e-6), \
        np.complex(-0.11094500303308243,0.20905585817734396), \
        np.complex(-1.1837814323046472e-6,3.8758497675466054e-7), \
        np.complex(0.1275629945870373,-0.1177987279989385), \
        np.complex(-2.690647673469878e-6,-3.6519804939862998e-6), \
        np.complex(0.12095531973074151,-0.19588735180644176), \
        np.complex(-2.6588791126371675e-6,-6.058292629669095e-7), \
        np.complex(0.052905370429015865,-0.0626791930782206), \
        np.complex(-1.6615538648519722e-7,6.756126951837809e-8), \
        np.complex(0.016378329200891946,-0.034743342821208854), \
        np.complex(4.408946495377283e-8,2.2826415255126898e-8), \
        np.complex(0.002765352838800482,-0.010624191776867055), \
        6.429253878486627e-8, \
        np.complex(0.00027095836439738105,-0.002684435917226972), \
        np.complex(1.1081202749445256e-8,-2.938812506852636e-8), \
        np.complex(-0.000055767533641099717,-0.000525444354381421), \
        np.complex(-1.0776974926155464e-8,-2.497769263148397e-8), \
        np.complex(-0.000024992489351114305,-0.00008178444317382933), \
        np.complex(-1.5079116121444066e-8,-2.0513760149701907e-8), \
        np.complex(-5.64035228941742e-6,-0.000010297667130821428), \
        np.complex(-1.488452012610573e-8,-1.7358623165948514e-8), \
        np.complex(-8.909884885392901e-7,-1.04267002748775e-6), \
        np.complex(-1.2056784102984098e-8,-1.2210951690230782e-8)], [0, \
        0.5871298855433338, \
        np.complex(-3.3729618710801137e-6,2.4152360811650373e-6), \
        np.complex(-0.5233926069798007,-0.13655786303346068), \
        np.complex(-4.623380373113224e-6,0.000010362902695259763), \
        np.complex(-0.17909656013941788,-0.11916639160269833), \
        np.complex(-3.399720873431807e-6,-7.125008373682292e-7), \
        np.complex(0.04072119358712736,-0.3719310475303641), \
        np.complex(-7.536125619789242e-6,1.885248226837573e-6), \
        np.complex(-0.11393851510585044,-0.3456924286310791), \
        np.complex(-2.3915763815197452e-6,-4.2406689395594674e-7), \
        np.complex(0.12820184730203607,0.0935942533049232), \
        np.complex(-1.5407293261691393e-6,-2.4673669087089514e-6), \
        np.complex(-0.012272903377715643,-0.13317144020065683), \
        np.complex(-1.1260776123106269e-6,-1.6865728072273087e-7), \
        np.complex(-0.01013345155253134,-0.0240812705564227), \
        np.complex(0.,-1.4163391111474348e-7), \
        np.complex(-0.003213070562510137,-0.012363639898516247), \
        np.complex(-1.0619280312362908e-8,-1.2021213613319027e-7), \
        np.complex(-0.002006756716685063,-0.0026636832583059812), \
        np.complex(0.,-4.509035934797572e-8), \
        np.complex(-0.00048585160444833446,-0.0005014735884977489), \
        np.complex(-1.2286988061034212e-8,-2.1199721851825594e-8), \
        np.complex(-0.00010897007463988193,-0.00007018240288615613), \
        np.complex(-1.2811279935244964e-8,-1.160553871672415e-8), \
        np.complex(-0.00001785800494916693,-6.603027186486886e-6), \
        -1.1639448324793031e-8, \
        np.complex(-2.4097385882316104e-6,-3.5223103057306496e-7), \
        -1.0792272866841885e-8, \
        np.complex(-2.597671478115077e-7,2.622928060603902e-8)]])

        all_num_codes = [states17, statesM, statesM2, statesP, statesP2]

        if idx is None:
            idx = torch.randint(len(all_num_codes), (1,)).item()

        probs = all_num_codes[idx]
        return probs

    def forward(self, hilbert_size, probs=None, idx=None, mu=None):
        """
        number code
        probs: custom prob for num state
        idx: index of specified num probs
        """
        if mu is None:
            mu = torch.randint(2, (1,)).item()

        state = Basis()(hilbert_size, 0)*0

        if probs is None:
            if idx is None: probs = self.get_random_num_prob()
            else: probs = self.get_random_num_prob(idx)

        for n, p in enumerate(probs[mu]):
            state += p*Basis()(hilbert_size, n)    
        
        psi = state.view(-1, 1)
        rho = psi*psi.adjoint()
        return rho*1/torch.trace(rho)
    
class BatchedFidelity(nn.Module):
    def __init__(self):
        super(BatchedFidelity, self).__init__()
        '''
        Fidelity score of 2 state.
        '''
    def sqrt_mat(self, A):
        assert len(A.shape) == 3, 'This function only take 3 dim input'
        for img in A:
            img = torch.tensor(scp.linalg.sqrtm(torch.squeeze(img).numpy()), dtype=torch.complex128)
        return A
        
    def forward(self, A, B):
        sqrtmA = self.sqrt_mat(A)
        temp = torch.matmul(torch.matmul(sqrtmA, B), sqrtmA)
        fidel = self.sqrt_mat(temp).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        fidel = fidel.real
        fidel = torch.sqrt(fidel)
        return fidel
    
class Reconstructor(nn.Module):
    '''
    Data reconstructor for quantum state tomography
    '''    
    def __init__(self, hilbert_size=32):
        super(Reconstructor, self).__init__()
        self.hilbert_size = hilbert_size
            
    def ConstructFockState(self, values):
        return Fock()(self.hilbert_size, n=int(round(values[0].real)))
    
    def ConstructCoherentState(self, values):
        return CoherentDM()(self.hilbert_size, alpha=values[0])
    
    def ConstructThermalState(self, values): 
        return Thermal()(self.hilbert_size, mean_photon_number=values[0].real)
    
    def ConstructNumState(self, values):
        idx = int(round(values[0].real))
        mu = int(round(values[1].real))
        if idx > 4 or idx <-5: idx = 0
        if mu > 2 or mu <-3: mu = 0
        return Num()(self.hilbert_size, idx=idx, mu=mu)
    
    def ConstructBinomialState(self, values):
        return Binomial()(self.hilbert_size, S=int(round(values[0].real)), N=int(round(values[1].real)), mu=int(round(values[2].real)))
    
    def ConstructCatState(self, values):
        return Cat()(self.hilbert_size, alpha=values[0], S=int(round(values[1].real)), mu=int(round(values[2].real)))
    
    def ConstructGKPState(self, values):
        return GKP()(self.hilbert_size, delta=values[0].real, mu=int(round(values[1].real)))
    
    def construct(self, 
                  label,
                  values):
      
        function_list = [self.ConstructFockState,
                         self.ConstructCoherentState,
                         self.ConstructThermalState,
                         self.ConstructNumState,
                         self.ConstructBinomialState,
                         self.ConstructCatState,
                         self.ConstructGKPState]
                
        return function_list[int(label.item())](values.tolist())
    
    def forward(self,
               labels,
               values):
        
        state_list = None
        
        if labels.dim() == 0:
            return self.construct(labels, values).type(torch.complex128)
        
        for label, value in zip(labels, values):
            state = self.construct(label, value)
            state = torch.unsqueeze(state, dim=0)
            if state_list is None:
                state_list = state
            else:
                state_list = torch.cat((state_list, state), 0)
        
        return state_list.type(torch.complex128)