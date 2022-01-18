from deepa2datasets.builder import Director
from deepa2datasets.nli_builder import NLIBuilder
from dataclasses import asdict

director = Director()
builder = NLIBuilder()
director.builder = builder

print("Standard basic product: ")
director.build_minimal_viable_product()
print(asdict(builder.product))

print("\n")

print("Standard full featured product: ")
director.build_full_featured_product()
builder.product.list_parts()

print("\n")
