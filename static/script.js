document.getElementById("prediction-form").addEventListener("submit", function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.prediction) {
            document.getElementById("result").innerText = data.prediction;
        } else if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        }
    })
    .catch(error => {
        document.getElementById("result").innerText = "An unexpected error occurred.";
        console.error("Error:", error);
    });
});
