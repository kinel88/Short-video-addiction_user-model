from kuairec_semantic_module import build_semantic_library

lib = build_semantic_library(
    caption_category_csv="kuairec_caption_category.csv",
    output_csv="video_semantic_library.csv",
    output_json="video_semantic_library.json",
)

print("Done.")
print(f"Number of videos: {len(lib)}")