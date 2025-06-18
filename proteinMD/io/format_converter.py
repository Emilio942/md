"""
Format Converter Utilities for ProteinMD

This module provides comprehensive format conversion utilities between
different molecular file formats, including batch processing capabilities.

Task 12.1: Multi-Format Support 🚀
Part of the comprehensive I/O format support implementation.

Author: GitHub Copilot
Date: January 2025
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
import argparse
from dataclasses import dataclass

from . import (
    MultiFormatIO, FormatType, FormatDetector,
    StructureData, TrajectoryData
)

logger = logging.getLogger(__name__)


@dataclass
class ConversionJob:
    """Represents a single format conversion job."""
    input_path: Path
    output_path: Path
    input_format: FormatType
    output_format: FormatType
    conversion_type: str  # 'structure' or 'trajectory'


class FormatConverter:
    """Comprehensive format conversion utility."""
    
    def __init__(self):
        """Initialize the format converter."""
        self.io_system = MultiFormatIO()
        self.supported_formats = self.io_system.get_supported_formats()
        
        logger.info("Format converter initialized")
        logger.info(f"Supported read formats: {len(self.supported_formats['read_structure'])}")
        logger.info(f"Supported write formats: {len(self.supported_formats['write_structure'])}")
    
    def convert_single_file(self, input_path: Union[str, Path], 
                           output_path: Union[str, Path],
                           input_format: Optional[FormatType] = None,
                           output_format: Optional[FormatType] = None,
                           force_overwrite: bool = False) -> bool:
        """
        Convert a single file between formats.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            input_format: Input format (auto-detected if None)
            output_format: Output format (auto-detected if None)
            force_overwrite: Overwrite existing files
            
        Returns:
            True if conversion successful, False otherwise
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Check input file exists
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return False
        
        # Check output file
        if output_path.exists() and not force_overwrite:
            logger.error(f"Output file exists (use force_overwrite=True): {output_path}")
            return False
        
        # Detect formats if not specified
        if input_format is None:
            input_format = FormatDetector.detect_format(input_path)
        
        if output_format is None:
            output_format = FormatDetector.detect_format(output_path)
        
        logger.info(f"Converting {input_path} ({input_format}) -> {output_path} ({output_format})")
        
        try:
            # Try structure conversion first
            try:
                self.io_system.convert_structure(input_path, output_path, output_format)
                logger.info("Structure conversion completed successfully")
                return True
            except Exception as struct_error:
                logger.debug(f"Structure conversion failed: {struct_error}")
                
                # Try trajectory conversion
                try:
                    self.io_system.convert_trajectory(input_path, output_path, output_format)
                    logger.info("Trajectory conversion completed successfully")
                    return True
                except Exception as traj_error:
                    logger.error(f"Both structure and trajectory conversion failed")
                    logger.error(f"Structure error: {struct_error}")
                    logger.error(f"Trajectory error: {traj_error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    def batch_convert(self, input_directory: Union[str, Path],
                     output_directory: Union[str, Path],
                     input_pattern: str = "*",
                     input_format: Optional[FormatType] = None,
                     output_format: FormatType = FormatType.PDB,
                     force_overwrite: bool = False,
                     create_subdirs: bool = True) -> Dict[str, bool]:
        """
        Batch convert multiple files.
        
        Args:
            input_directory: Directory containing input files
            output_directory: Directory for output files
            input_pattern: File pattern to match (e.g., "*.xyz")
            input_format: Input format (auto-detected if None)
            output_format: Output format
            force_overwrite: Overwrite existing files
            create_subdirs: Create subdirectories in output
            
        Returns:
            Dictionary mapping file paths to conversion success status
        """
        input_dir = Path(input_directory)
        output_dir = Path(output_directory)
        
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return {}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        input_files = list(input_dir.glob(input_pattern))
        
        if not input_files:
            logger.warning(f"No files found matching pattern: {input_pattern}")
            return {}
        
        logger.info(f"Found {len(input_files)} files for batch conversion")
        
        results = {}
        
        for input_file in input_files:
            # Determine output file path
            if create_subdirs:
                # Preserve directory structure
                relative_path = input_file.relative_to(input_dir)
                output_file = output_dir / relative_path.with_suffix(f".{output_format.value}")
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Flat output structure
                output_file = output_dir / f"{input_file.stem}.{output_format.value}"
            
            # Convert file
            success = self.convert_single_file(
                input_file, output_file, input_format, output_format, force_overwrite
            )
            
            results[str(input_file)] = success
            
            if success:
                logger.info(f"✅ Converted: {input_file.name}")
            else:
                logger.error(f"❌ Failed: {input_file.name}")
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"Batch conversion completed: {successful}/{total} files successful")
        
        return results
    
    def create_conversion_pipeline(self, jobs: List[ConversionJob],
                                 continue_on_error: bool = True) -> Dict[str, bool]:
        """
        Execute a pipeline of conversion jobs.
        
        Args:
            jobs: List of conversion jobs to execute
            continue_on_error: Continue pipeline if individual jobs fail
            
        Returns:
            Dictionary mapping job descriptions to success status
        """
        logger.info(f"Starting conversion pipeline with {len(jobs)} jobs")
        
        results = {}
        
        for i, job in enumerate(jobs, 1):
            job_desc = f"Job {i}: {job.input_path.name} -> {job.output_path.name}"
            logger.info(f"Executing {job_desc}")
            
            try:
                if job.conversion_type == 'structure':
                    self.io_system.convert_structure(
                        job.input_path, job.output_path, job.output_format
                    )
                elif job.conversion_type == 'trajectory':
                    self.io_system.convert_trajectory(
                        job.input_path, job.output_path, job.output_format
                    )
                else:
                    # Auto-detect conversion type
                    success = self.convert_single_file(
                        job.input_path, job.output_path, 
                        job.input_format, job.output_format, force_overwrite=True
                    )
                    results[job_desc] = success
                    continue
                
                results[job_desc] = True
                logger.info(f"✅ {job_desc} completed")
                
            except Exception as e:
                logger.error(f"❌ {job_desc} failed: {e}")
                results[job_desc] = False
                
                if not continue_on_error:
                    logger.error("Pipeline stopped due to error")
                    break
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"Pipeline completed: {successful}/{total} jobs successful")
        
        return results
    
    def validate_conversion(self, original_file: Union[str, Path],
                           converted_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a file conversion by comparing key properties.
        
        Args:
            original_file: Path to original file
            converted_file: Path to converted file
            
        Returns:
            Dictionary with validation results
        """
        original_file = Path(original_file)
        converted_file = Path(converted_file)
        
        validation = {
            'original_exists': original_file.exists(),
            'converted_exists': converted_file.exists(),
            'validation_errors': [],
            'properties_match': False,
            'conversion_valid': False
        }
        
        if not validation['original_exists']:
            validation['validation_errors'].append("Original file not found")
            return validation
        
        if not validation['converted_exists']:
            validation['validation_errors'].append("Converted file not found")
            return validation
        
        try:
            # Read both files
            original_info = self.io_system.validate_file(original_file)
            converted_info = self.io_system.validate_file(converted_file)
            
            # Compare key properties
            if original_info.get('is_structure') and converted_info.get('is_structure'):
                orig_atoms = original_info.get('n_atoms', 0)
                conv_atoms = converted_info.get('n_atoms', 0)
                
                if orig_atoms == conv_atoms:
                    validation['properties_match'] = True
                else:
                    validation['validation_errors'].append(
                        f"Atom count mismatch: {orig_atoms} -> {conv_atoms}"
                    )
            
            elif original_info.get('is_trajectory') and converted_info.get('is_trajectory'):
                orig_frames = original_info.get('n_frames', 0)
                conv_frames = converted_info.get('n_frames', 0)
                
                if orig_frames == conv_frames:
                    validation['properties_match'] = True
                else:
                    validation['validation_errors'].append(
                        f"Frame count mismatch: {orig_frames} -> {conv_frames}"
                    )
            
            validation['conversion_valid'] = (
                validation['properties_match'] and 
                len(validation['validation_errors']) == 0
            )
            
        except Exception as e:
            validation['validation_errors'].append(f"Validation error: {e}")
        
        return validation
    
    def get_conversion_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        Get a matrix showing possible conversions between formats.
        
        Returns:
            Nested dictionary showing conversion possibilities
        """
        read_formats = self.supported_formats['read_structure']
        write_formats = self.supported_formats['write_structure']
        
        matrix = {}
        
        for read_fmt in read_formats:
            matrix[read_fmt.value] = {}
            for write_fmt in write_formats:
                matrix[read_fmt.value][write_fmt.value] = True
        
        return matrix
    
    def suggest_conversion_path(self, input_format: FormatType, 
                               target_format: FormatType) -> List[FormatType]:
        """
        Suggest conversion path between formats (direct or via intermediate format).
        
        Args:
            input_format: Source format
            target_format: Target format
            
        Returns:
            List of formats representing conversion path
        """
        # Check direct conversion
        if (input_format in self.supported_formats['read_structure'] and
            target_format in self.supported_formats['write_structure']):
            return [input_format, target_format]
        
        # Check via intermediate format (e.g., PDB as universal intermediate)
        if FormatType.PDB in self.supported_formats['write_structure']:
            if (input_format in self.supported_formats['read_structure'] and
                target_format in self.supported_formats['write_structure']):
                return [input_format, FormatType.PDB, target_format]
        
        # No conversion path found
        return []


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface for format conversion."""
    parser = argparse.ArgumentParser(
        description="ProteinMD Multi-Format Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python -m proteinMD.io.format_converter input.pdb output.xyz
  
  # Batch convert directory
  python -m proteinMD.io.format_converter -b input_dir/ output_dir/ --pattern "*.pdb" --output-format xyz
  
  # Validate conversion
  python -m proteinMD.io.format_converter --validate original.pdb converted.xyz
"""
    )
    
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('output', nargs='?', help='Output file or directory')
    
    parser.add_argument('-b', '--batch', action='store_true',
                       help='Batch convert directory')
    parser.add_argument('--pattern', default='*',
                       help='File pattern for batch conversion')
    parser.add_argument('--input-format', type=str,
                       help='Input format (auto-detected if not specified)')
    parser.add_argument('--output-format', type=str,
                       help='Output format (auto-detected if not specified)')
    parser.add_argument('-f', '--force', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--validate', action='store_true',
                       help='Validate conversion')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    return parser


def main():
    """Main CLI function."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    converter = FormatConverter()
    
    if args.validate:
        if not args.output:
            print("Error: Validation requires both input and output files")
            return 1
        
        validation = converter.validate_conversion(args.input, args.output)
        
        print(f"Validation Results:")
        print(f"Original exists: {validation['original_exists']}")
        print(f"Converted exists: {validation['converted_exists']}")
        print(f"Properties match: {validation['properties_match']}")
        print(f"Conversion valid: {validation['conversion_valid']}")
        
        if validation['validation_errors']:
            print("Errors:")
            for error in validation['validation_errors']:
                print(f"  - {error}")
        
        return 0 if validation['conversion_valid'] else 1
    
    if args.batch:
        if not args.output:
            print("Error: Batch conversion requires output directory")
            return 1
        
        # Parse formats
        input_format = FormatType(args.input_format) if args.input_format else None
        output_format = FormatType(args.output_format) if args.output_format else FormatType.PDB
        
        results = converter.batch_convert(
            args.input, args.output,
            input_pattern=args.pattern,
            input_format=input_format,
            output_format=output_format,
            force_overwrite=args.force
        )
        
        successful = sum(results.values())
        total = len(results)
        print(f"Batch conversion completed: {successful}/{total} files successful")
        
        return 0 if successful == total else 1
    
    else:
        # Single file conversion
        if not args.output:
            print("Error: Single file conversion requires output file")
            return 1
        
        # Parse formats
        input_format = FormatType(args.input_format) if args.input_format else None
        output_format = FormatType(args.output_format) if args.output_format else None
        
        success = converter.convert_single_file(
            args.input, args.output,
            input_format=input_format,
            output_format=output_format,
            force_overwrite=args.force
        )
        
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
