import torch
import torch.nn.functional as F

class ADDMNIST_SL(torch.nn.Module):
    def __init__(self, loss, logic, args) -> None:
        super().__init__()
        self.base_loss = loss
        self.logic = logic
        # Worlds-queries matrix
        if args.task == 'addition':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
            self.nr_classes = 19
        elif args.task == 'product':
            self.n_facts = 10 if not args.dataset == 'restrictedmnist' else 5
            self.nr_classes = 37
        elif args.task == 'multiop':
            self.n_facts = 5
            self.nr_classes = 3

    def compute_query(self, query, worlds_prob):
            """Computes query probability given the worlds probability P(w)."""
            # Select the column of w_q matrix corresponding to the current query
            w_q = self.logic[:, query]
            # Compute query probability by summing the probability of all the worlds where the query is true
            query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
            return query_prob

    def forward(self, out_dict, args):
        
        loss, losses = self.base_loss(out_dict, args)
        
        # load from dict
        Y = out_dict['LABELS']
        pCs = out_dict['pCS']

        prob_digit1, prob_digit2 = pCs[:, 0, :], pCs[:, 1,:]

        # Compute worlds probability P(w) (the two digits values are independent)
        Z_1 = prob_digit1[..., None]
        Z_2 = prob_digit2[:, None, :]

        probs = Z_1.multiply(Z_2)

        worlds_prob = probs.reshape(-1, self.n_facts * self.n_facts)
        
        # Compute query probability P(q)
        query_prob = torch.zeros(size=(len(probs), self.nr_classes), device=probs.device)

        for i in range(self.nr_classes):
            query = i
            query_prob[:,i] = self.compute_query(query, worlds_prob).view(-1)

        # add a small offset
        query_prob += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob, dim=-1, keepdim=True)
        query_prob = query_prob / Z
        
        sl = F.nll_loss(query_prob.log(), Y.to(torch.long), reduction='mean')
                
        losses.update({'sl': sl.item()})

        return loss + args.w_sl*sl, losses