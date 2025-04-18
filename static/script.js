document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
  
    const formData = {
      Area: parseFloat(document.getElementById('area').value),
      Bedrooms: parseInt(document.getElementById('bedrooms').value),
      Bathrooms: parseFloat(document.getElementById('bathrooms').value),
      Country: document.getElementById('country').value,
      City: document.getElementById('city').value
    };
  
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Calculating...';
  
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });
  
      const result = await response.json();
  
      if (result.status === 'success') {
        resultDiv.textContent = `Predicted Price: ﷼ ${result.prediction.toLocaleString(undefined, { maximumFractionDigits: 0 })} SAR`;
      } 
      else {
        resultDiv.textContent = `Error: ${result.message}`;
      }
    } catch (error) {
      resultDiv.textContent = `Connection error: ${error.message}`;
    }
  });
  