import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, rdDistGeom
from rdkit.Chem.rdMolDescriptors import CalcTPSA
import requests
import math
from scipy.spatial.distance import cdist

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========================================================================================
# PROTEIN STRUCTURE HANDLER
# ========================================================================================
class ProteinStructure:
    """Handles protein structure loading and analysis from PDB"""
    def __init__(self, pdb_id=None, pdb_data=None):
        self.pdb_id = pdb_id
        self.atoms = []
        self.residues = []
        self.binding_site = None
        self.coordinates = None
        if pdb_id:
            self.load_from_pdb_id(pdb_id)
        elif pdb_data:
            self.parse_pdb_data(pdb_data)
    
    def load_from_pdb_id(self, pdb_id):
        """Load protein structure from PDB ID"""
        try:
            url = f"https://files.rcsb.org/download/ {pdb_id.upper()}.pdb"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self.parse_pdb_data(response.text)
            print(f"✅ Loaded protein {pdb_id} from PDB")
        except Exception as e:
            print(f"❌ Could not load PDB {pdb_id}: {e}")
            # Use mock protein data for demonstration
            self.create_mock_protein()
    
    def create_mock_protein(self):
        """Create a mock protein structure for demonstration"""
        print("🔧 Creating mock protein structure...")
        mock_residues = [
            {'name': 'ARG', 'chain': 'A', 'resnum': 123, 'coords': np.array([10.0, 15.0, 20.0])},
            {'name': 'ASP', 'chain': 'A', 'resnum': 156, 'coords': np.array([12.0, 16.0, 18.0])},
            {'name': 'PHE', 'chain': 'A', 'resnum': 201, 'coords': np.array([8.0, 14.0, 22.0])},
            {'name': 'TYR', 'chain': 'A', 'resnum': 234, 'coords': np.array([11.0, 13.0, 19.0])},
            {'name': 'HIS', 'chain': 'A', 'resnum': 267, 'coords': np.array([9.0, 17.0, 21.0])},
        ]
        self.residues = mock_residues
        self.binding_site = {
            'center': np.array([10.0, 15.0, 20.0]),
            'radius': 8.0,
            'key_residues': ['ARG123', 'ASP156', 'PHE201', 'TYR234', 'HIS267']
        }
        self.coordinates = np.array([res['coords'] for res in self.residues])
    
    def parse_pdb_data(self, pdb_content):
        """Parse PDB file content"""
        atoms = []
        residues = {}
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM'):
                atom_data = {
                    'atom_name': line[12:16].strip(),
                    'residue_name': line[17:20].strip(),
                    'chain': line[21:22],
                    'residue_num': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': line[76:78].strip()
                }
                atoms.append(atom_data)
                res_key = f"{atom_data['chain']}{atom_data['residue_num']}"
                if res_key not in residues:
                    residues[res_key] = {
                        'name': atom_data['residue_name'],
                        'chain': atom_data['chain'],
                        'resnum': atom_data['residue_num'],
                        'atoms': [],
                        'coords': []
                    }
                residues[res_key]['atoms'].append(atom_data)
                residues[res_key]['coords'].append([atom_data['x'], atom_data['y'], atom_data['z']])
        self.atoms = atoms
        self.residues = list(residues.values())
        
        for residue in self.residues:
            residue['coords'] = np.mean(residue['coords'], axis=0)
        self.coordinates = np.array([res['coords'] for res in self.residues])
        
        if len(self.residues) > 0:
            self.identify_binding_site()
    
    def identify_binding_site(self, ligand_coords=None):
        """Identify potential binding site"""
        if ligand_coords is None:
            center = np.mean(self.coordinates, axis=0)
        else:
            center = np.array(ligand_coords)
            
        distances = np.linalg.norm(self.coordinates - center, axis=1)
        binding_residues = [i for i, d in enumerate(distances) if d < 10.0]
        
        self.binding_site = {
            'center': center,
            'radius': 10.0,
            'key_residues': [f"{self.residues[i]['name']}{self.residues[i]['resnum']}" 
                           for i in binding_residues[:10]]
        }

# ========================================================================================
# MOLECULAR DOCKING SIMULATOR
# ========================================================================================
class MolecularDocking:
    """Simulates molecular docking between ligand and protein"""
    def __init__(self, protein_structure):
        self.protein = protein_structure
        self.interaction_weights = {
            'hydrophobic': 1.0,
            'hydrogen_bond': 2.0,
            'electrostatic': 1.5,
            'van_der_waals': 0.8,
            'pi_stacking': 1.2
        }
    
    def dock_molecule(self, smiles):
        """Perform simplified docking calculation"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'binding_affinity': -999, 'interaction_score': 0, 'pose_confidence': 0}
        
        mol = Chem.AddHs(mol)
        success = rdDistGeom.EmbedMolecule(mol, randomSeed=SEED)
        if success != 0:
            return {'binding_affinity': -999, 'interaction_score': 0, 'pose_confidence': 0}
        
        props = self._calculate_ligand_properties(mol)
        binding_score = self._calculate_binding_score(mol, props)
        pose_confidence = min(1.0, max(0.0, (binding_score + 10) / 15))
        
        return {
            'binding_affinity': binding_score,
            'interaction_score': self._calculate_interaction_score(mol, props),
            'pose_confidence': pose_confidence,
            'ligand_properties': props
        }
    
    def _calculate_ligand_properties(self, mol):
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            'tpsa': CalcTPSA(mol)
        }
    
    def _calculate_binding_score(self, mol, props):
        score = 0.0
        
        # Size complementarity
        mw = props['molecular_weight']
        if 200 <= mw <= 500:
            score += 3.0
        else:
            score -= abs(mw - 350) / 100
        
        # Hydrophobic interactions
        logp = props['logp']
        if 1 <= logp <= 4:
            score += 2.0 * self.interaction_weights['hydrophobic']
        else:
            score -= abs(logp - 2.5) * 0.5
        
        # Hydrogen bonding potential
        hb_potential = (props['num_hbd'] + props['num_hba']) / 2
        if 2 <= hb_potential <= 6:
            score += hb_potential * self.interaction_weights['hydrogen_bond']
        
        # Electrostatic interactions
        if abs(props['formal_charge']) <= 1:
            score += 1.0 * self.interaction_weights['electrostatic']
        else:
            score -= abs(props['formal_charge']) * 0.5
        
        # Aromatic stacking
        if props['aromatic_rings'] > 0:
            score += min(props['aromatic_rings'], 3) * self.interaction_weights['pi_stacking']
        
        # Flexibility penalty
        flexibility_penalty = max(0, props['num_rotatable_bonds'] - 5) * 0.3
        score -= flexibility_penalty
        
        # TPSA contribution
        tpsa = props['tpsa']
        if 20 <= tpsa <= 130:
            score += 1.0
        else:
            score -= abs(tpsa - 75) / 50
        
        # Protein-specific bonuses
        if self.protein.binding_site:
            score += self._calculate_protein_specific_score(props)
        
        return score
    
    def _calculate_protein_specific_score(self, props):
        bonus = 0.0
        key_residues = self.protein.binding_site.get('key_residues', [])
        
        for residue in key_residues:
            if residue.startswith('ARG') or residue.startswith('LYS'):
                if props['formal_charge'] < 0:
                    bonus += 1.5
            elif residue.startswith('ASP') or residue.startswith('GLU'):
                if props['formal_charge'] > 0:
                    bonus += 1.5
            elif residue.startswith('PHE') or residue.startswith('TYR') or residue.startswith('TRP'):
                if props['aromatic_rings'] > 0:
                    bonus += 1.0
            elif residue.startswith('SER') or residue.startswith('THR') or residue.startswith('HIS'):
                if props['num_hbd'] > 0 or props['num_hba'] > 0:
                    bonus += 0.8
        
        return min(bonus, 5.0)
    
    def _calculate_interaction_score(self, mol, props):
        interaction_score = 0.0
        interaction_score += props['num_hbd'] * self.interaction_weights['hydrogen_bond']
        interaction_score += props['num_hba'] * self.interaction_weights['hydrogen_bond']
        interaction_score += props['aromatic_rings'] * self.interaction_weights['pi_stacking']
        interaction_score += min(props['logp'], 4) * self.interaction_weights['hydrophobic']
        return interaction_score

# ========================================================================================
# ENHANCED SMILES VOCABULARY
# ========================================================================================
class SMILESVocabulary:
    """Enhanced vocabulary for SMILES-Strings with better tokenization"""
    def __init__(self):
        self.tokens = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I',
            'c', 'n', 'o', 's', 'p',  
            '(', ')', '[', ']', '=', '#', '-', '+', '.',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '@', '@@', 'H', '/', '\\',
            '[NH]', '[OH]', '[CH]', '[nH]', '[O-]', '[N+]',
            '<PAD>', '<START>', '<END>', '<UNK>'
        ]
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
    
    def tokenize(self, smiles):
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i] == '[':
                bracket_end = smiles.find(']', i)
                if bracket_end != -1:
                    bracket_token = smiles[i:bracket_end+1]
                    if bracket_token in self.token_to_id:
                        tokens.append(bracket_token)
                        i = bracket_end + 1
                        continue
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.token_to_id:
                    tokens.append(two_char)
                    i += 2
                    continue
            char = smiles[i]
            if char in self.token_to_id:
                tokens.append(char)
            else:
                tokens.append('<UNK>')
            i += 1
        return tokens
    
    def encode(self, smiles, max_length=100):
        tokens = ['<START>'] + self.tokenize(smiles) + ['<END>']
        ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
        if len(ids) < max_length:
            ids.extend([self.token_to_id['<PAD>']] * (max_length - len(ids)))
        else:
            ids = ids[:max_length]
        return ids
    
    def decode(self, ids):
        tokens = [self.id_to_token.get(id, self.id_to_token['<UNK>']) for id in ids]
        tokens = [t for t in tokens if t not in ['<PAD>', '<START>', '<END>', '<UNK>']]
        return ''.join(tokens)

# ========================================================================================
# ENHANCED MOLECULAR EVALUATOR WITH PROTEIN INTERACTION
# ========================================================================================
class ProteinLigandEvaluator:
    """Enhanced evaluator including protein-ligand interactions"""
    def __init__(self, protein_structure, docking_simulator):
        self.protein = protein_structure
        self.docking = docking_simulator
    
    def calculate_properties(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            props = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': CalcTPSA(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
                'valid': True
            }
            
            try:
                props['qed'] = Descriptors.qed(mol)
            except:
                props['qed'] = 0.0
            
            lipinski_violations = sum([
                props['molecular_weight'] > 500,
                props['logp'] > 5,
                props['num_hbd'] > 5,
                props['num_hba'] > 10
            ])
            props['lipinski_violations'] = lipinski_violations
            
            docking_results = self.docking.dock_molecule(smiles)
            props.update(docking_results)
            
            return props
        except Exception as e:
            print(f"Error calculating properties for {smiles}: {e}")
            return None
    
    def calculate_reward(self, smiles, target_properties=None):
        props = self.calculate_properties(smiles)
        if props is None:
            return -15.0
        
        reward = 0.0
        drug_likeness_reward = self._calculate_drug_likeness_reward(props, target_properties)
        reward += drug_likeness_reward * 0.3
        binding_reward = self._calculate_binding_reward(props)
        reward += binding_reward * 0.5
        interaction_reward = self._calculate_interaction_reward(props)
        reward += interaction_reward * 0.2
        return reward
    
    def _calculate_drug_likeness_reward(self, props, target_properties):
        if target_properties is None:
            target_properties = {
                'molecular_weight': (200, 500),
                'logp': (0, 5),
                'tpsa': (20, 130),
                'qed': (0.4, 1.0),
                'lipinski_violations': (0, 1)
            }
        
        reward = 0.0
        mw = props['molecular_weight']
        if target_properties['molecular_weight'][0] <= mw <= target_properties['molecular_weight'][1]:
            reward += 3.0
        else:
            penalty = abs(mw - np.mean(target_properties['molecular_weight'])) / 100
            reward -= min(penalty, 3.0)
        
        logp = props['logp']
        if target_properties['logp'][0] <= logp <= target_properties['logp'][1]:
            reward += 3.0
        else:
            penalty = abs(logp - np.mean(target_properties['logp']))
            reward -= min(penalty, 3.0)
        
        tpsa = props['tpsa']
        if target_properties['tpsa'][0] <= tpsa <= target_properties['tpsa'][1]:
            reward += 2.0
        else:
            penalty = abs(tpsa - np.mean(target_properties['tpsa'])) / 50
            reward -= min(penalty, 2.0)
        
        reward += props['qed'] * 4.0
        reward -= props['lipinski_violations'] * 2.0
        return reward
    
    def _calculate_binding_reward(self, props):
        binding_affinity = props.get('binding_affinity', -999)
        if binding_affinity <= -10:
            return -5.0
        
        if binding_affinity >= 10:
            return 10.0
        elif binding_affinity >= 5:
            return binding_affinity
        elif binding_affinity >= 0:
            return binding_affinity * 0.5
        else:
            return binding_affinity * 0.3
    
    def _calculate_interaction_reward(self, props):
        interaction_score = props.get('interaction_score', 0)
        pose_confidence = props.get('pose_confidence', 0)
        interaction_reward = interaction_score * 0.7 + pose_confidence * 3.0
        return min(interaction_reward, 5.0)

# ========================================================================================
# ENHANCED SMILES GENERATOR WITH ATTENTION
# ========================================================================================
class EnhancedSMILESGenerator(nn.Module):
    """Enhanced SMILES generator with attention mechanism"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = lstm_out + attended
        output = self.dropout(output)
        output = self.output_layer(output)
        return output, hidden
    
    def generate_smiles(self, vocab, max_length=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            current_token = torch.tensor([[vocab.token_to_id['<START>']]])
            hidden = None
            generated_ids = []
            for step in range(max_length):
                output, hidden = self.forward(current_token, hidden)
                logits = output[0, -1, :] / temperature
                
                if step > max_length * 0.8:
                    logits = logits / 1.5
                
                logits[vocab.token_to_id['<PAD>']] = -float('inf')
                logits[vocab.token_to_id['<START>']] = -float('inf')
                
                if step < 5:
                    for token in ['C', 'N', 'O', 'c', 'n', 'o']:
                        if token in vocab.token_to_id:
                            logits[vocab.token_to_id[token]] += 0.5
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == vocab.token_to_id['<END>']:
                    break
                
                generated_ids.append(next_token)
                current_token = torch.tensor([[next_token]])
            
            return vocab.decode(generated_ids)

# ========================================================================================
# ENHANCED RL TRAINER WITH PROTEIN FEEDBACK
# ========================================================================================
class ProteinRLTrainer:
    """Enhanced RL trainer with protein-specific feedback"""
    def __init__(self, generator, vocab, evaluator, lr=1e-4):
        self.generator = generator
        self.vocab = vocab
        self.evaluator = evaluator
        self.optimizer = optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.8)
        
        self.rewards_history = []
        self.losses_history = []
        self.binding_affinity_history = []
        self.validity_history = []
        self.baseline = None
    
    def train_episode(self, num_molecules=32, baseline_alpha=0.9):
        self.generator.train()
        molecules_data = []
        log_probs = []
        rewards = []
        
        for _ in range(num_molecules):
            smiles, log_prob = self._generate_with_log_prob()
            reward = self.evaluator.calculate_reward(smiles)
            props = self.evaluator.calculate_properties(smiles)
            
            molecules_data.append({
                'smiles': smiles,
                'reward': reward,
                'properties': props,
                'log_prob': log_prob
            })
            log_probs.append(log_prob)
            rewards.append(reward)
        
        current_avg_reward = np.mean(rewards) if rewards else 0
        if self.baseline is None:
            self.baseline = current_avg_reward
        else:
            self.baseline = baseline_alpha * self.baseline + (1 - baseline_alpha) * current_avg_reward
        
        if len(rewards) > 0:
            policy_loss = 0
            for log_prob, reward in zip(log_probs, rewards):
                advantage = reward - self.baseline
                policy_loss -= log_prob * advantage
            policy_loss = policy_loss / len(rewards)
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
            policy_loss = torch.tensor(0.0)
        
        self.scheduler.step()
        
        valid_molecules = [m for m in molecules_data if m['properties'] is not None]
        binding_affinities = [m['properties']['binding_affinity'] for m in valid_molecules 
                            if m['properties']['binding_affinity'] > -999]
        
        stats = {
            'avg_reward': current_avg_reward,
            'max_reward': max(rewards) if rewards else 0,
            'loss': policy_loss.item(),
            'molecules_data': molecules_data,
            'validity_rate': len(valid_molecules) / len(molecules_data),
            'avg_binding_affinity': np.mean(binding_affinities) if binding_affinities else -999,
            'max_binding_affinity': max(binding_affinities) if binding_affinities else -999,
            'baseline': self.baseline,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self.rewards_history.append(current_avg_reward)
        self.losses_history.append(policy_loss.item())
        self.binding_affinity_history.append(stats['avg_binding_affinity'])
        self.validity_history.append(stats['validity_rate'])
        
        return stats
    
    def _generate_with_log_prob(self, max_length=50, temperature=1.0):
        self.generator.train()
        current_token = torch.tensor([[self.vocab.token_to_id['<START>']]])
        hidden = None
        generated_ids = []
        log_prob_sum = 0
        
        for _ in range(max_length):
            output, hidden = self.generator(current_token, hidden)
            logits = output[0, -1, :] / temperature
            
            logits[self.vocab.token_to_id['<PAD>']] = -float('inf')
            logits[self.vocab.token_to_id['<START>']] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            next_token = dist.sample()
            log_prob_sum += dist.log_prob(next_token)
            
            if next_token.item() == self.vocab.token_to_id['<END>']:
                break
            
            generated_ids.append(next_token.item())
            current_token = torch.tensor([[next_token.item()]])
        
        smiles = self.vocab.decode(generated_ids)
        return smiles, log_prob_sum
    










def main():
    # Initialisiere Komponenten
    protein = ProteinStructure(pdb_id="1ABC")  # Beispiel-PDB-ID
    docking_simulator = MolecularDocking(protein)
    evaluator = ProteinLigandEvaluator(protein, docking_simulator)
    vocab = SMILESVocabulary()
    generator = EnhancedSMILESGenerator(vocab.vocab_size)
    trainer = ProteinRLTrainer(generator, vocab, evaluator)
    
    # Trainingsschleife
    num_episodes = 100
    for episode in range(num_episodes):
        stats = trainer.train_episode()
        print(f"Episode {episode+1}: Avg Reward={stats['avg_reward']:.2f}, "
              f"Avg Binding Affinity={stats['avg_binding_affinity']:.2f}")
    
    # Generiere einige Moleküle nach dem Training
    print("\n🧪 Generierte Moleküle nach dem Training:")
    for _ in range(5):
        smiles, _ = trainer._generate_with_log_prob()
        props = evaluator.calculate_properties(smiles)
        print(f"SMILES: {smiles}")
        print(f"Binding Affinity: {props['binding_affinity']:.2f}")
        print(f"Drug-likeness Score: {props['qed']:.2f}\n")

if __name__ == "__main__":
    main()