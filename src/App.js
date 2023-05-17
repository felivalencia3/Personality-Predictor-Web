import React, { useState, useRef, useEffect } from "react"; // import useRef and useEffect hooks

import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css"; // import bootstrap css
import "./App.css"; // import custom css

function App() {
  // Define state variables for name, text, result and loading
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false); // add loading state

  // Define a reference variable for the image container element
  const imageContainerRef = useRef(null);

  // Define an effect hook to scroll to the image container when the result state changes
  useEffect(() => {
    if (result) {
      // If there is a result, scroll to the image container element
      imageContainerRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [result]); // Only run this effect when result changes

  // Define a function to handle form submission
  const handleSubmit = async (e) => {
    // Prevent default browser behavior
    e.preventDefault();
    // Set the loading state to true
    setLoading(true);
    // Send a post request to the API with the name and text as data
    try {
      const response = await axios.post("/api/predict", { name, text });
      // Set the result state to the response data
      setResult(response.data);
    } catch (error) {
      // Handle any errors
      console.error(error);
    }
    // Set the loading state to false
    setLoading(false);
  };

  // Define a function to handle name input change
  const handleNameChange = (e) => {
    // Set the name state to the input value
    setName(e.target.value);
  };

  // Define a function to handle text input change
  const handleTextChange = (e) => {
    // Set the text state to the input value
    setText(e.target.value);
  };

  return (
    <div className="page-wrapper bg-dark p-t-100 p-b-50">
      <div className="wrapper wrapper--w900">
        <div className="card card-6">
          <div className="card-heading">
            <h2 className="title">Predict MBTI Personality Type:</h2>
          </div>
          <div className="card-body">
            <form onSubmit={handleSubmit}>
              <div className="form-row">
                <div className="name">Name</div>
                <div className="value">
                  <input
                    className="input--style-6"
                    type="text"
                    name="full_name"
                    value={name}
                    onChange={handleNameChange}
                    required
                  />
                </div>
              </div>
              <div className="form-row">
                <div className="name">Text</div>
                <div className="value">
                  <div className="input-group">
                    <textarea
                      className="textarea--style-6"
                      name="message"
                      value={text}
                      onChange={handleTextChange}
                      placeholder="A Bit of Your Writing..."
                      required
                    ></textarea>
                  </div>
                </div>
              </div>
              <div className="card-footer">
                <button className="btn btn--radius-2 btn--blue-2" type="submit">
                  Send Application
                </button>
                {loading && <p className="loading-text">Loading...</p>}
              </div>
            </form>
          </div>
          {result && (
            <div className="image-container" ref={imageContainerRef}>
              <img src={`data:image/png;base64,${result.image}`} alt="graph" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
