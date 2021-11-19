

def ekf(self, u, y, h, l, step, P=None, Q=None, R=None):
        self.initPQR(P, Q, R) 
        # Compute NN jacobian 
        D = self._affine_dot2(self.W[1], self.dsig(l))  # flatten = 3394560 
        d_size = self.ny * self.ny
        
        # As H is an arbitrary matrix, so, extract by flattening
        H = D.flatten()[:d_size].reshape(self.ny, self.ny) 
        
        one_array = np.ones((l.shape), order='C') 
        l_concat = np.concatenate((l, one_array), axis=1) 
        
        W0_len = len(self.W[0])
        W1_len = len(self.W[1]) 
        
        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R 
        K = self.P.dot(H.T).dot(npl.inv(S)) 
        self.P = self.P - np.dot(K, H.dot(self.P)) 
        
        # Update weight estimates and covariance
        y = np.resize(y, (self.ny, self.ny))
        self.P = self.P - np.dot(K, H.dot(self.P)) 
        dW = step*K.dot(y - H) 
        self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)
        
        if np.any(self.Q): self.Q_nonzero = True
        else: self.Q_nonzero = False
            
        if self.Q_nonzero: self.P = self.P + self.Q
        return self.P, self.R