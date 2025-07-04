#!/usr/bin/env python3
"""
Memory leak fixes for existing simulation files.
This script patches existing simulation code to fix identified memory leaks.
"""

import os
import shutil
import re
from pathlib import Path


def create_memory_fix_patches():
    """Create patches for common memory leak patterns."""
    
    # Core simulation memory fixes (for proteinMD/core/simulation.py)
    core_simulation_patches = [
        {
            'description': 'Fix unbounded trajectory storage',
            'pattern': r'self\.trajectory = \[\]',
            'replacement': 'from collections import deque\n        self.trajectory = deque(maxlen=1000)  # Bounded trajectory storage'
        },
        {
            'description': 'Fix unbounded energy storage',
            'pattern': r'self\.energies = {\'kinetic\': \[\], \'potential\': \[\], \'total\': \[\]}',
            'replacement': '''self.energies = {
            'kinetic': deque(maxlen=10000), 
            'potential': deque(maxlen=10000), 
            'total': deque(maxlen=10000)
        }'''
        },
        {
            'description': 'Add periodic garbage collection',
            'pattern': r'def step_simulation\(self\):',
            'replacement': '''def step_simulation(self):
        # Periodic garbage collection for long runs
        if hasattr(self, 'step_count') and self.step_count % 1000 == 0:
            import gc
            gc.collect()'''
        }
    ]
    
    return core_simulation_patches


def fix_trajectory_storage_pattern(file_path: str):
    """Fix the common trajectory storage memory leak pattern."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if file already has memory fixes
        if 'deque' in content and 'maxlen' in content:
            print(f"  ✅ {file_path} already has memory optimizations")
            return False
            
        original_content = content
        modified = False
        
        # Fix 1: Replace unbounded trajectory lists with bounded deques
        patterns_to_fix = [
            (r'self\.trajectories = \[\]', 'from collections import deque\n        self.trajectories = deque(maxlen=1000)'),
            (r'self\.trajectory = \[\]', 'from collections import deque\n        self.trajectory = deque(maxlen=1000)'),
            (r'self\.energies = \[\]', 'from collections import deque\n        self.energies = deque(maxlen=5000)'),
            (r'self\.temperatures = \[\]', 'from collections import deque\n        self.temperatures = deque(maxlen=5000)'),
            (r'self\.trajectories\.append\(', 'self.trajectories.append('),  # This stays the same but now bounded
        ]
        
        for pattern, replacement in patterns_to_fix:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                print(f"  🔧 Fixed trajectory storage pattern in {file_path}")
        
        # Fix 2: Add garbage collection in step functions
        step_pattern = r'(def step_simulation\(self\):.*?)([\n\r])'
        if re.search(step_pattern, content, re.DOTALL):
            def add_gc_check(match):
                return match.group(1) + '\n        # Periodic garbage collection\n        if hasattr(self, "step_count"):\n            self.step_count += 1\n            if self.step_count % 1000 == 0:\n                import gc\n                gc.collect()\n        else:\n            self.step_count = 1' + match.group(2)
            
            content = re.sub(step_pattern, add_gc_check, content, flags=re.DOTALL)
            modified = True
            print(f"  🔧 Added garbage collection to {file_path}")
        
        # Fix 3: Add deque import if not present
        if 'from collections import deque' not in content and 'deque(' in content:
            # Add import at the top
            import_section = content.split('\n')
            for i, line in enumerate(import_section):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    import_section.insert(i, 'from collections import deque')
                    break
            content = '\n'.join(import_section)
            modified = True
            print(f"  🔧 Added deque import to {file_path}")
        
        if modified:
            # Create backup
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)
            print(f"  💾 Created backup: {backup_path}")
            
            # Write fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Applied memory fixes to {file_path}")
            return True
        else:
            print(f"  ➖ No memory leak patterns found in {file_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False


def fix_matplotlib_memory_leaks(file_path: str):
    """Fix matplotlib memory leaks in animation code."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Fix matplotlib object cleanup
        patterns_to_fix = [
            # Fix quiver object removal
            (r'viz_elements\[\'quiver\'\]\.remove\(\)', 
             '''try:
                try:
                viz_elements['quiver'].remove()
            except:
                pass
            except:
                pass'''),
            
            # Fix collections cleanup
            (r'ax\.collections\.clear\(\)',
             '''# Proper cleanup of matplotlib collections
            for collection in ax.collections[:]:
                try:
                    collection.remove()
                except:
                    pass
            # Proper cleanup of matplotlib collections
            for collection in ax.collections[:]:
                try:
                    collection.remove()
                except:
                    pass
            ax.collections.clear()'''),
            
            # Add proper figure cleanup
            (r'plt\.show\(\)',
             '''plt.show()
        # Cleanup after animation
        plt.close('all')
        import gc
        gc.collect()
        # Cleanup after animation
        plt.close('all')
        import gc
        gc.collect()''')
        ]
        
        for pattern, replacement in patterns_to_fix:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                modified = True
                print(f"  🔧 Fixed matplotlib cleanup pattern in {file_path}")
        
        if modified:
            # Create backup
            backup_path = file_path + '.mpl_backup'
            shutil.copy2(file_path, backup_path)
            
            # Write fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"  ✅ Applied matplotlib fixes to {file_path}")
            return True
        else:
            print(f"  ➖ No matplotlib memory leak patterns found in {file_path}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False


def apply_memory_fixes_to_project():
    """Apply memory leak fixes to all simulation files in the project."""
    print("APPLYING MEMORY LEAK FIXES TO PROJECT FILES")
    print("="*60)
    
    # Find all Python files in the project
    project_dir = Path("/home/emilio/Documents/ai/md")
    python_files = list(project_dir.glob("**/*.py"))
    
    # Filter out test files and our new optimized files
    simulation_files = []
    for file_path in python_files:
        if (not file_path.name.startswith('test_') and 
            not file_path.name.startswith('memory_') and
            not file_path.name.startswith('optimized_') and
            file_path.name.endswith('.py')):
            simulation_files.append(file_path)
    
    print(f"Found {len(simulation_files)} simulation files to check:")
    for file_path in simulation_files:
        print(f"  - {file_path}")
    
    print("\nApplying memory leak fixes...")
    
    fixes_applied = 0
    
    for file_path in simulation_files:
        print(f"\nProcessing: {file_path}")
        
        # Apply trajectory storage fixes
        if fix_trajectory_storage_pattern(str(file_path)):
            fixes_applied += 1
            
        # Apply matplotlib fixes
        if fix_matplotlib_memory_leaks(str(file_path)):
            fixes_applied += 1
    
    print(f"\n{'='*60}")
    print(f"MEMORY LEAK FIXES SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(simulation_files)}")
    print(f"Fixes applied: {fixes_applied}")
    
    if fixes_applied > 0:
        print("✅ Memory leak fixes have been applied to the codebase!")
        print("📋 Backup files created with .backup and .mpl_backup extensions")
        print("🧪 Run test_memory_leaks.py to validate the fixes")
    else:
        print("ℹ️  No memory leak patterns found or all files already optimized")
    
    return fixes_applied


def create_memory_optimization_guide():
    """Create a guide for memory optimization best practices."""
    guide_content = """# MEMORY OPTIMIZATION GUIDE FOR MOLECULAR DYNAMICS SIMULATIONS

## Common Memory Leak Patterns and Fixes

### 1. Unbounded List Growth
❌ **Problem:**
```python
self.trajectories = []
self.energies = []
# These lists grow indefinitely during simulation
```

✅ **Solution:**
```python
from collections import deque
self.trajectories = deque(maxlen=1000)  # Keep only last 1000 frames
self.energies = deque(maxlen=5000)      # Keep only last 5000 energy values
```

### 2. Matplotlib Object Leaks
❌ **Problem:**
```python
# Proper cleanup of matplotlib collections
            for collection in ax.collections[:]:
                try:
                    collection.remove()
                except:
                    pass
            ax.collections.clear()  # Doesn't properly remove objects
quiver.remove()         # May fail silently
```

✅ **Solution:**
```python
# Proper cleanup of matplotlib collections
for collection in ax.collections[:]:
    try:
        collection.remove()
    except:
        pass
# Proper cleanup of matplotlib collections
            for collection in ax.collections[:]:
                try:
                    collection.remove()
                except:
                    pass
            ax.collections.clear()

# Safe object removal
try:
    quiver.remove()
except:
    pass
```

### 3. Large Array Allocations
❌ **Problem:**
```python
field_grid = np.zeros((100, 100, 100))  # Large arrays created repeatedly
```

✅ **Solution:**
```python
# Pre-allocate and reuse buffers
if not hasattr(self, '_field_buffer'):
    self._field_buffer = np.zeros((50, 50, 50))  # Smaller, reusable buffer
```

### 4. Missing Garbage Collection
❌ **Problem:**
```python
for step in range(10000):
    simulate_step()  # No memory management
```

✅ **Solution:**
```python
for step in range(10000):
    simulate_step()
    if step % 1000 == 0:
        import gc
        gc.collect()  # Periodic cleanup
```

## Memory-Efficient Simulation Design

### Use Bounded Storage
- Limit trajectory history to reasonable lengths (1000-5000 frames)
- Use deque with maxlen for automatic size management
- Store only essential data for analysis

### Optimize Neighbor Lists
- Update neighbor lists only when necessary (every 10-20 steps)
- Use cutoff distances to limit calculations
- Pre-allocate neighbor list storage

### Manage Visualization Objects
- Clean up matplotlib objects after each frame
- Close figures when done with animations
- Use plt.ioff() for non-interactive plots

### Monitor Memory Usage
- Use the MemoryProfiler class for continuous monitoring
- Set memory growth thresholds (< 1-2 MB/min for stable simulations)
- Profile long-running simulations regularly

## Validation Checklist

✅ Simulation runs 10,000+ steps with constant memory usage
✅ No continuous memory growth over 30+ minutes
✅ Memory profiling shows growth rate < 1 MB/min
✅ Matplotlib animations don't accumulate objects
✅ Trajectory storage is bounded and efficient

## Tools and Scripts

- `memory_profiler.py` - Real-time memory monitoring
- `optimized_simulation.py` - Memory-optimized MD simulation class
- `test_memory_leaks.py` - Comprehensive memory leak testing
- `fix_memory_leaks.py` - Automated fixing of common patterns

## Performance Tips

1. Use numpy operations instead of Python loops
2. Pre-allocate arrays when possible
3. Limit visualization resolution for real-time displays
4. Use appropriate data types (float32 vs float64)
5. Implement lazy evaluation for expensive calculations

Remember: A well-optimized simulation should maintain constant memory usage
regardless of simulation length!
"""
    
    guide_path = "/home/emilio/Documents/ai/md/MEMORY_OPTIMIZATION_GUIDE.md"
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    print(f"📚 Memory optimization guide created: {guide_path}")


if __name__ == "__main__":
    print("MOLECULAR DYNAMICS MEMORY LEAK FIXING TOOL")
    print("="*60)
    
    # Apply fixes to project files
    fixes_applied = apply_memory_fixes_to_project()
    
    # Create optimization guide
    create_memory_optimization_guide()
    
    print(f"\n🎯 Memory leak fixing complete!")
    
    if fixes_applied > 0:
        print("✅ Fixes have been applied to simulation files")
        print("🧪 Next step: Run 'python test_memory_leaks.py' to validate fixes")
    else:
        print("ℹ️  No fixes needed - code appears to be already optimized")
        print("🧪 You can still run 'python test_memory_leaks.py' to verify memory usage")
    
    print("\n📚 Memory optimization guide available at: MEMORY_OPTIMIZATION_GUIDE.md")
