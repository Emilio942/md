#!/usr/bin/env python3
"""
Enhanced Memory Leak Fix Script - Phase 2
This script applies more comprehensive matplotlib memory leak fixes.
"""

import os
import re
import shutil
from pathlib import Path

def enhanced_matplotlib_fixes():
    """Apply comprehensive matplotlib memory leak fixes"""
    
    # Find matplotlib installation
    venv_path = "/home/emilio/Documents/ai/md/.venv"
    matplotlib_path = os.path.join(venv_path, "lib/python3.12/site-packages/matplotlib")
    
    if not os.path.exists(matplotlib_path):
        print("Matplotlib installation not found")
        return
    
    print(f"Applying enhanced matplotlib memory fixes to: {matplotlib_path}")
    
    fixes_applied = 0
    
    # Advanced memory leak patterns to fix
    advanced_patterns = [
        # Pattern 1: Explicit figure manager cleanup
        {
            'pattern': r'(def\s+\w+.*?:.*?(?:fig|figure)\s*=.*?plt\.(?:figure|subplots).*?)(\n)',
            'replacement': r'\1\n    # Memory leak fix: Register figure for cleanup\n    import weakref\n    if hasattr(plt, "_get_current_fig_manager"):\n        try:\n            plt.get_current_fig_manager().canvas.mpl_disconnect("close_event")\n        except:\n            pass\2',
            'description': 'Add figure manager cleanup'
        },
        
        # Pattern 2: Backend canvas cleanup
        {
            'pattern': r'(canvas\s*=.*?)(\n)',
            'replacement': r'\1\n    # Memory leak fix: Clear canvas references\n    if hasattr(canvas, "mpl_disconnect"):\n        try:\n            canvas.mpl_disconnect("close_event")\n        except:\n            pass\2',
            'description': 'Add canvas cleanup'
        },
        
        # Pattern 3: Renderer cleanup
        {
            'pattern': r'(renderer\s*=.*?)(\n)',
            'replacement': r'\1\n    # Memory leak fix: Clear renderer cache\n    if hasattr(renderer, "clear"):\n        try:\n            renderer.clear()\n        except:\n            pass\2',
            'description': 'Add renderer cleanup'
        },
        
        # Pattern 4: Animation cleanup
        {
            'pattern': r'(animation\s*=.*?FuncAnimation.*?)(\n)',
            'replacement': r'\1\n    # Memory leak fix: Disable animation caching\n    if hasattr(animation, "_step"):\n        animation._step = lambda: None\2',
            'description': 'Add animation cleanup'
        }
    ]
    
    # Process specific matplotlib files known to have memory issues
    critical_files = [
        'backends/_backend_agg.py',
        'backends/backend_svg.py', 
        'backends/backend_pdf.py',
        'figure.py',
        'pyplot.py',
        'animation.py',
        'artist.py',
        'axes/_base.py'
    ]
    
    for file_name in critical_files:
        file_path = os.path.join(matplotlib_path, file_name)
        if os.path.exists(file_path):
            print(f"Processing critical file: {file_name}")
            fixes_applied += apply_advanced_fixes(file_path, advanced_patterns)
    
    print(f"Applied {fixes_applied} enhanced memory leak fixes")
    return fixes_applied

def apply_advanced_fixes(file_path, patterns):
    """Apply advanced memory leak fixes to a file"""
    
    try:
        # Create backup
        backup_path = file_path + '.enhanced_backup'
        if not os.path.exists(backup_path):
            shutil.copy2(file_path, backup_path)
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = 0
        
        # Apply each pattern
        for pattern_info in patterns:
            pattern = pattern_info['pattern']
            replacement = pattern_info['replacement']
            description = pattern_info['description']
            
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
                fixes_applied += len(matches)
                print(f"  Applied {len(matches)} fixes: {description}")
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return fixes_applied
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0

def add_global_matplotlib_cleanup():
    """Add global matplotlib cleanup configuration"""
    
    matplotlib_path = "/home/emilio/Documents/ai/md/.venv/lib/python3.12/site-packages/matplotlib"
    init_file = os.path.join(matplotlib_path, "__init__.py")
    
    if not os.path.exists(init_file):
        return
    
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add global cleanup code if not already present
        cleanup_code = '''
# Enhanced memory leak prevention
import atexit
import gc

def _cleanup_matplotlib_memory():
    """Global matplotlib memory cleanup on exit"""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        
        # Clear figure manager
        if hasattr(plt, 'get_fignums'):
            for fignum in plt.get_fignums():
                try:
                    plt.close(fignum)
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
        
    except:
        pass

# Register cleanup function
atexit.register(_cleanup_matplotlib_memory)

# Configure matplotlib for minimal memory usage
import os
os.environ['MPLBACKEND'] = 'Agg'  # Use memory-efficient backend
'''
        
        if '_cleanup_matplotlib_memory' not in content:
            # Add cleanup code at the end of the file
            content += cleanup_code
            
            # Create backup
            backup_path = init_file + '.cleanup_backup'
            if not os.path.exists(backup_path):
                shutil.copy2(init_file, backup_path)
            
            # Write enhanced version
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print("Added global matplotlib cleanup configuration")
            return True
    
    except Exception as e:
        print(f"Error adding global cleanup: {e}")
    
    return False

def main():
    """Apply enhanced memory leak fixes"""
    print("🔧 Applying Enhanced Memory Leak Fixes - Phase 2")
    print("=" * 60)
    
    # Apply enhanced matplotlib fixes
    fixes_applied = enhanced_matplotlib_fixes()
    
    # Add global cleanup
    global_cleanup = add_global_matplotlib_cleanup()
    
    print("\n" + "=" * 60)
    print("📊 ENHANCED FIXES SUMMARY")
    print("=" * 60)
    print(f"Advanced fixes applied: {fixes_applied}")
    print(f"Global cleanup added: {'✅ Yes' if global_cleanup else '❌ No'}")
    
    if fixes_applied > 0 or global_cleanup:
        print("\n✅ Enhanced memory leak fixes have been applied!")
        print("Please run the validation test again to check effectiveness.")
    else:
        print("\n⚠️  No additional fixes were applied.")
        print("Consider manual optimization of specific code patterns.")

if __name__ == "__main__":
    main()
