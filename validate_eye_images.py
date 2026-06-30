import os
import csv

from app import is_eye_image


def main():
    dataset_root = "Dataset"
    output_csv = "eye_validation_report.csv"

    # open CSV up front and write header; we'll append rows periodically
    # use utf-8 to handle emoji/error messages
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "split", "class", "is_valid", "error_message"])

        buffer = []
        count = 0
        t_cnt = 0

        # walk through the Train/Test splits and class subfolders
        for split in ("Train", "Test"):
            for cls in ("Cataract", "Normal"):
                folder = os.path.join(dataset_root, split, cls)
                if not os.path.isdir(folder):
                    print(f"⚠️  Skipping missing folder: {folder}")
                    continue
                for fname in os.listdir(folder):
                    fpath = os.path.join(folder, fname)
                    if not os.path.isfile(fpath):
                        continue
                    # run the 5-layer robust is_eye_image validator
                    is_valid, error_msg = is_eye_image(fpath)
                    t_cnt += 1
                    status = "✓ VALID" if is_valid else "✗ INVALID"
                    print(f"{status} | {fname} | {t_cnt} | {error_msg if not is_valid else ''}")
                    buffer.append((fpath, split, cls, is_valid, error_msg))
                    count += 1

                    # flush every 10 rows
                    if count % 5 == 0:
                        writer.writerows(buffer)
                        buffer.clear()
        # write any remaining rows
        if buffer:
            writer.writerows(buffer)

    print(f"\n✓ Finished validation. Report written to {output_csv}")
    print(f"Total images processed: {t_cnt}")


if __name__ == "__main__":
    main()
