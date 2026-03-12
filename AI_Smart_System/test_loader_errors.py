from io import BytesIO
import environment_monitor as em

# simulate empty file
try:
    em.load_environment_data(BytesIO(b""))
except Exception as e:
    print("caught error as expected:", e)

# simulate file with wrong headers
wrong = "foo,bar\n1,2".encode()
try:
    em.load_environment_data(BytesIO(wrong))
except Exception as e:
    print("caught wrong header error:", e)
