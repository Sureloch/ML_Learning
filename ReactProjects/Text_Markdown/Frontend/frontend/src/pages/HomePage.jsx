import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'

function HomePage ({token, id}){
    const [notes , setNotes] = useState([]);
    const [title, setTitle] = useState("");
    const [body, setBody] = useState("");
    const [updatedTitle, updateTitle] =useState("");
    const [updatedBody, updateBody] =useState("");
    const [certainNote, updateCertainNote] = useState(null);
    const newNote = () => {setShowForm(true)}
    const [showForm, setShowForm] = useState(false);
    const handleDeletion = (e) => {
        e.preventDefault();
        deleteNote();
    }
    const handleSubmit = (e) => {
        e.preventDefault();
        addNote();
    }
    const handleUpdateSubmit = (e) => {
        console.log(certainNote)
        e.preventDefault();
        updateNote();
    }
    useEffect(() =>
    {

        fetchNotes();
        return () => {
        };
    },[]);
    const deleteNote = async () =>{
        const response = await fetch(`http://localhost:8000/delete_note/${certainNote.id}`, {
                    method : 'DELETE',
                headers: {
                'token': token,
                'Accept': 'application/json',
                'Content-Type': 'application/json'
                },
                })
                    if (response.ok) {
                    updateCertainNote(null)
                    fetchNotes()
                }
    }
    const updateNote = async () =>{
        const response = await fetch(`http://localhost:8000/update_note/${certainNote.id}`, {
            method : 'PATCH',
          headers: {
          'token': token,
          'Accept': 'application/json',
          'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            title : updatedTitle,
            body : updatedBody
          })
        })
            if (response.ok) {
            updateCertainNote(null)
            fetchNotes()
        }
    }
    async function fetchNotes() {
            const response = await fetch(`http://localhost:8000/all_notes`, {
            headers: {
            'token': token
            }
            })
            const data = await response.json()
            setNotes(data)
            }
    const addNote = async () =>{
        const response = await fetch(`http://localhost:8000/new_note`, {
            method : 'POST',
          headers: {
          'token': token,
          'Accept': 'application/json',
          'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            title : title,
            body : body
          })
        })
        if (response.ok) {
            setShowForm(false)
            fetchNotes()
        }
    }
    return(
        <div>
        <button onClick={newNote}>Add Note</button>
        {showForm && <div>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="Enter Title"
                />
                <textarea
                    
                    type = "text"
                    value={body}
                    onChange={(e) => setBody(e.target.value)}
                    placeholder="Write your note here..."
                />
                <button type="submit">Sumbit</button>
                </form>  
            </div>}
        {certainNote != null && <div>
            <form onSubmit={handleUpdateSubmit}>
                <input
                    type="text"
                    value={updatedTitle}
                    onChange={(e) => updateTitle(e.target.value)}
                />
                <textarea
                    type = "text"
                    value={updatedBody}
                    onChange={(e) => updateBody(e.target.value)}
                />
                <ReactMarkdown>{updatedBody}</ReactMarkdown>                
                <button type="submit">Sumbit</button>
                </form>
            <form onSubmit={handleDeletion}>
                <button type="submit">Delete</button>
                </form>  
        </div>}
        {notes.map(note => (
        <div key={note.id} onClick={() => {        
        updateCertainNote(note)
        updateBody(note.body)
        updateTitle(note.title)}
        }  style={{cursor: 'pointer'}}>
            <h1>{note.title}</h1>
        </div>
    ))}
        </div>
    )
}
export default HomePage