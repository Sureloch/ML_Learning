import { use, useState } from "react"
const inputStyle = "border border-gray-300 rounded p-2 w-full"
const buttonStyle = "border border-gray-300 rounded"
function GetClicksFromAlias() {
  const [alias, setAlias] = useState('');
  const [result, setResult] = useState(null);


  const fetchTest = async () => {
    const response = await fetch(`http://localhost:8000/${alias}/stats`)
    const data = await response.json()
    setResult(data.clicks)
    console.log(result)
  }
  const handleSubmit = (e) => {
    e.preventDefault();
    fetchTest();
  }
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={alias}
        onChange={(e) => setAlias(e.target.value)}
        className={inputStyle}
        placeholder="Enter the Alias to get clicks from"
      />
      <button type="submit"
        className={buttonStyle}
      >Sumbit</button>
      <p>
        {result != null && <div>The number of clicks to {alias} is {result}</div>}
        {result == null && alias != '' && <div>Alias not found</div>}
      </p>
    </form>

  );
}

function SetUpURLFromAlias() {
  const [newalias, setNewAlias] = useState('');
  const [oldURL, setURL] = useState('');
  const [showModal, setShowModal] = useState(false)

  const shortenURL = async () => {
    const response = await fetch('http://localhost:8000/shorten/',
      {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          alias: newalias,
          original_url: oldURL
        })
      });
    const data = await response.json()
    if (response.ok) {
      setShowModal(true)
    }




  }

  const handleSubmit = (e) => {
    e.preventDefault();
    shortenURL();
  }

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={oldURL}
        onChange={(e) => setURL(e.target.value)}
        placeholder="Existing URL"
        className={inputStyle}
      />

      <input
        type="text"
        value={newalias}
        onChange={(e) => setNewAlias(e.target.value)}
        placeholder="New Alias"
        className={inputStyle}
      />

      <button type="submit"
        className={buttonStyle}>
        Sumbit</button>
      <p></p>
      {showModal && <div>Successfully redirected old url to {newalias}</div>}
    </form>

  );
}
function DeleteAlias() {
  const [requestedAlias, setAlias] = useState('');
  const [oldURL, setOldURL] = useState('');
  const [modal, setModal] = useState(false);

  const deleteAlias = async () => {
    const response = await fetch(`http://localhost:8000/${requestedAlias}/remove_alias`, { method: 'DELETE' })
    const data = await response.json()
    if (response.ok) {
      setModal(true)
    } else {
      setModal(false)
    }

  }
  const handleSubmit = (e) => {
    e.preventDefault();
    deleteAlias();
  }
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={requestedAlias}
        onChange={(e) => setAlias(e.target.value)}
        placeholder="Enter the Alias to delete"
        className={inputStyle}
      />
      <button type="submit"
        className={buttonStyle}>Sumbit</button>
      <p>
        {modal && <div>Successfully deleted {requestedAlias}</div>}
        {!modal && requestedAlias != '' && <div>Alias not found</div>}
      </p>
    </form>

  );

}


function GoToAlias() {
  const [alias, setAlias] = useState('');


  const fetchTest = () => {
    window.location.href = `http://localhost:8000/${alias}`

  }
  const handleSubmit = (e) => {
    e.preventDefault();
    fetchTest();
  }
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={alias}
        onChange={(e) => setAlias(e.target.value)}
        placeholder="Enter the alias to go to"
        className={inputStyle}
      />
      <button type="submit" className={buttonStyle}>Go</button>
    </form>

  );
}
function App() {
  return (
    <div class="text-black mx-auto px-20">
      <h1 class="text-2xl font-bold">Shorten URL</h1>
      <p class="mt-4">This website is used to shorten URL, see how often the URL is visited, and delete said alias.</p>
      <SetUpURLFromAlias />
      <GoToAlias />
      <GetClicksFromAlias />
      <DeleteAlias />
    </div>
    /*<div class="container mx-auto px-4">
       
    </div>*/

  )
}
export default App