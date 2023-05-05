import logo from './logo.svg';
import './App.css';

function App() {

  function handleButtonClick() {
    fetch('/api/run-script')
      .then(response => response.json())
      .then(result => {
        console.log(result)
      })
      .catch(error => {
        console.error(error)
      })
  }
  

  return (
    <div className="App">
      <input></input>
      <button onClick={handleButtonClick}>Submit</button>
    </div>
  );
}

export default App;
