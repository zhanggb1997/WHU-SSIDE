'''
Project    : RSDeploy
FileName   : RemoveGeoHeadTransf .py
CreateTime : 2025/6/19 
=======================
@CopyRight : WHU-星光团队
@Author    : 弓长广文武
@Contact   : zhanggb1997@163.com
@Content   : # 实现内容 #
'''
# !/usr/bin/env python3
"""
Automatically remove geotransformation information from all TIFF files in a directory.
The original files will be backed up before modification.
"""

from osgeo import gdal
import os
import glob
import shutil
import argparse
from datetime import datetime
import sys

# Suppress GDAL warnings
gdal.PushErrorHandler('CPLQuietErrorHandler')


def has_geotransformation(tif_path):
    """
    Check if a TIFF file has geotransformation information.

    Args:
        tif_path: Path to the TIFF file

    Returns:
        bool: True if file has geotransformation, False otherwise
    """
    try:
        ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
        if ds is None:
            return False

        # Check geotransform (default is 0,1,0,0,0,1)
        gt = ds.GetGeoTransform()
        has_gt = gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # Check projection
        proj = ds.GetProjection()
        has_proj = bool(proj and proj.strip())

        # Check GCPs
        has_gcps = ds.GetGCPCount() > 0

        ds = None

        return has_gt or has_proj or has_gcps

    except Exception:
        return False


def remove_geotransformation(input_tif, backup=True):
    """
    Remove geotransformation information from a GeoTIFF file in place.

    Args:
        input_tif: Path to the input GeoTIFF file
        backup: Whether to create a backup of the original file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create backup if requested
        if backup:
            backup_path = f"{input_tif}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(input_tif, backup_path)
            print(f"  Created backup: {os.path.basename(backup_path)}")

        # Create temporary output file
        temp_output = input_tif + ".temp"

        # Open the source dataset
        src_ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
        if src_ds is None:
            print(f"  ERROR: Unable to open {input_tif}")
            return False

        # Get raster properties
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        bands = src_ds.RasterCount

        # Get data type of the first band
        band = src_ds.GetRasterBand(1)
        dtype = band.DataType

        # Create output dataset without geotransformation
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(temp_output, width, height, bands, dtype)

        if dst_ds is None:
            print(f"  ERROR: Unable to create temporary file")
            src_ds = None
            return False

        # Copy raster data band by band
        for i in range(1, bands + 1):
            src_band = src_ds.GetRasterBand(i)
            dst_band = dst_ds.GetRasterBand(i)

            # Read and write data
            data = src_band.ReadAsArray()
            dst_band.WriteArray(data)

            # Copy band metadata
            band_metadata = src_band.GetMetadata()
            if band_metadata:
                dst_band.SetMetadata(band_metadata)

            # Copy color interpretation
            dst_band.SetColorInterpretation(src_band.GetColorInterpretation())

            # Copy color table if exists
            color_table = src_band.GetColorTable()
            if color_table:
                dst_band.SetColorTable(color_table)

            # Copy nodata value if exists
            nodata = src_band.GetNoDataValue()
            if nodata is not None:
                dst_band.SetNoDataValue(nodata)

            # Flush cache
            dst_band.FlushCache()

        # Copy general metadata (excluding geotransformation-related)
        metadata = src_ds.GetMetadata()
        if metadata:
            filtered_metadata = {}
            geo_keywords = ['GEOTIFF', 'PROJECTION', 'DATUM', 'SPHEROID', 'PRIMEM',
                            'UNIT', 'AUTHORITY', 'TOWGS84', 'EXTENSION', 'AREA_OR_POINT']

            for key, value in metadata.items():
                if not any(keyword in key.upper() for keyword in geo_keywords):
                    filtered_metadata[key] = value

            if filtered_metadata:
                dst_ds.SetMetadata(filtered_metadata)

        # Close datasets
        dst_ds = None
        src_ds = None

        # Replace original file with processed one
        os.remove(input_tif)
        os.rename(temp_output, input_tif)

        return True

    except Exception as e:
        print(f"  ERROR: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False


def process_directory(directory, pattern="*.tif", backup=True, recursive=False):
    """
    Process all TIFF files in a directory and remove geotransformation information.

    Args:
        directory: Directory containing TIFF files
        pattern: File pattern to match
        backup: Whether to create backups
        recursive: Whether to search subdirectories
    """
    print(f"\nProcessing directory: {directory}")
    print(f"Backup enabled: {backup}")
    print(f"Recursive search: {recursive}")
    print("-" * 50)

    # Find all TIFF files
    if recursive:
        tif_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tif_files.append(os.path.join(root, file))
    else:
        tif_files = glob.glob(os.path.join(directory, "*.tif"))
        tif_files.extend(glob.glob(os.path.join(directory, "*.tiff")))
        tif_files.extend(glob.glob(os.path.join(directory, "*.TIF")))
        tif_files.extend(glob.glob(os.path.join(directory, "*.TIFF")))

    # Remove duplicates
    tif_files = list(set(tif_files))

    if not tif_files:
        print("No TIFF files found in the directory.")
        return

    print(f"Found {len(tif_files)} TIFF file(s)")
    print("-" * 50)

    # Process each file
    processed = 0
    skipped = 0
    failed = 0

    for tif_file in sorted(tif_files):
        rel_path = os.path.relpath(tif_file, directory)
        print(f"\nChecking: {rel_path}")

        if has_geotransformation(tif_file):
            print("  Has geotransformation - removing...")
            if remove_geotransformation(tif_file, backup):
                print("  ✓ Successfully removed geotransformation")
                processed += 1
            else:
                print("  ✗ Failed to remove geotransformation")
                failed += 1
        else:
            print("  No geotransformation found - skipping")
            skipped += 1

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  Total files found: {len(tif_files)}")
    print(f"  Processed: {processed}")
    print(f"  Skipped (no geo info): {skipped}")
    print(f"  Failed: {failed}")
    print("=" * 50)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Remove geotransformation information from TIFF files in a directory."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="/home/dshare/01Data/3DDisp/LiDAR/America/images/GF7_DLC_W87.6_N42.0_20210804_L1A0000774072",
        help="Directory containing TIFF files (default: current directory)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for geotransformation without removing"
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        sys.exit(1)

    # Convert to absolute path
    directory = os.path.abspath(args.directory)

    if args.check_only:
        # Only check mode
        print(f"\nChecking directory: {directory}")
        print(f"Recursive search: {args.recursive}")
        print("-" * 50)

        if args.recursive:
            tif_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.tif', '.tiff')):
                        tif_files.append(os.path.join(root, file))
        else:
            tif_files = glob.glob(os.path.join(directory, "*.tif"))
            tif_files.extend(glob.glob(os.path.join(directory, "*.tiff")))
            tif_files.extend(glob.glob(os.path.join(directory, "*.TIF")))
            tif_files.extend(glob.glob(os.path.join(directory, "*.TIFF")))

        tif_files = list(set(tif_files))

        if not tif_files:
            print("No TIFF files found.")
            return

        with_geo = 0
        without_geo = 0

        for tif_file in sorted(tif_files):
            rel_path = os.path.relpath(tif_file, directory)
            if has_geotransformation(tif_file):
                print(f"[GEO] {rel_path}")
                with_geo += 1
            else:
                print(f"[NO GEO] {rel_path}")
                without_geo += 1

        print("\n" + "-" * 50)
        print(f"Total files: {len(tif_files)}")
        print(f"With geotransformation: {with_geo}")
        print(f"Without geotransformation: {without_geo}")
    else:
        # Process mode
        process_directory(directory, backup=not args.no_backup, recursive=args.recursive)


if __name__ == "__main__":
    main()
