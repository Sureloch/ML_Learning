import { use, useState } from "react";
import Box from '@mui/material/Box';

//IN APP const[state, setState] = useState(null)

function Modal({ onClose, children }) {
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <div className="bg-white rounded-lg p-6">
        {children}
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  )
}
function ShowQuestions(){
  const [questions, setQuestions] = useState([])
  const getQuestion = async() => {
    const response = await fetch(`http://localhost:8000/all_questions`)
    const data = await response.json()
    setQuestions(data)
    console.log(questions)
  }
}


function CreateQuestion(props) {
  const [newQuestion, setNewQuestion] = useState('');
  const [showForm, setShowForm] = useState(false)
  const [answers, setAnswers] = useState([
    { text: '', isCorrect: true },
    { text: '', isCorrect: false },
    { text: '', isCorrect: false },
    { text: '', isCorrect: false },
  ])
  const initalAnswers = [
    { text: '', isCorrect: true },
    { text: '', isCorrect: false },
    { text: '', isCorrect: false },
    { text: '', isCorrect: false },
  ]
  const [questionNumber, setQuestionNumber] = useState(1)
  
  const inputAnswer = async () => {
    for(let i = 0; i < answers.length; i++){
      console.log(answers[i])
      const response =  await fetch(`http://localhost:8000/answer`, {
        method: 'POST',
        headers : {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: answers[i].text,
        question_id : questionNumber,
        is_correct : answers[i].isCorrect
      })

      })
    };
    handleCloseForm();
  }
  const inputQuestion = async () => {
    const response = await fetch(`http://localhost:8000/question`, {
      method: "POST",
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: newQuestion
      })
    });
    const data = await response.json()
  }
  const handleCloseForm = () => {
    console.log(questionNumber)
    props.onAddQuestion({ question: newQuestion, answers: answers })
    setNewQuestion('')
    setAnswers(initalAnswers)
    //ShowQuestions()
    setShowForm(false)

  }
  const handleSubmit = (e) => {
    e.preventDefault();
    inputQuestion();
    inputAnswer();
    setQuestionNumber(prev => prev + 1)

  }
  const handleChange = (index, e) => {
    const copy = [...answers]
    copy[index].text = e.target.value
    setAnswers(copy)
  }
  const handleRadio = (index, e) => {
    const updated = answers.map((item, i) => ({
      ...item,
      isCorrect: i === index
    }))
    setAnswers(updated)
  }
  return (
  <div>
    {!showForm && (
      <button
        onClick={() => setShowForm(true)}
        style={{
          padding: '10px 20px',
          fontSize: '14px',
          fontWeight: '600',
          background: '#4f46e5',
          color: '#fff',
          border: 'none',
          borderRadius: '8px',
          cursor: 'pointer',
        }}
      >
        + Add Question
      </button>
    )}

    {showForm && (
      <Modal onClose={() => setShowForm(false)}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          padding: '28px',
          background: '#fff',
          borderRadius: '12px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.12)',
          minWidth: '360px',
        }}>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: '#1e1b4b' }}>
            New Question
          </h2>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            <input
              type="text"
              value={newQuestion}
              onChange={(e) => setNewQuestion(e.target.value)}
              placeholder="Enter question"
              style={{
                padding: '10px 14px',
                fontSize: '14px',
                border: '1.5px solid #e0e0e0',
                borderRadius: '8px',
                outline: 'none',
                width: '100%',
                boxSizing: 'border-box',
              }}
            />

            <p style={{ margin: 0, fontSize: '13px', fontWeight: '600', color: '#6b7280' }}>
              ANSWERS — select the correct one
            </p>

            {answers.map((answer, index) => (
              <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <input
                  type="radio"
                  name="correct"
                  checked={answer.isCorrect}
                  onChange={() => handleRadio(index)}
                  style={{ accentColor: '#4f46e5', width: '16px', height: '16px', cursor: 'pointer' }}
                />
                <input
                  type="text"
                  value={answer.text}
                  placeholder={`Answer ${index + 1}`}
                  onChange={(e) => handleChange(index, e)}
                  style={{
                    flex: 1,
                    padding: '8px 12px',
                    fontSize: '14px',
                    border: `1.5px solid ${answer.isCorrect ? '#4f46e5' : '#e0e0e0'}`,
                    borderRadius: '8px',
                    outline: 'none',
                    background: answer.isCorrect ? '#f5f3ff' : '#fff',
                    boxSizing: 'border-box',
                  }}
                />
              </div>
            ))}

            <button
              type="submit"
              style={{
                marginTop: '6px',
                padding: '11px',
                fontSize: '14px',
                fontWeight: '600',
                background: '#4f46e5',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
              }}
            >
              Submit
            </button>
          </form>
        </div>
      </Modal>
    )}
  </div>
);
}




function EnterUser({ onSuccess }) {
  const [new_name, setName] = useState('');
  const [result, setResult] = useState(null);

  const inputUser = async () => {
    const response = await fetch(`http://localhost:8000/create_user`,
      {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          name: new_name
        })
      });
    const data = await response.json()
    if (response.ok)
      onSuccess(data.id)
  }
  const handleSubmit = (e) => {
    e.preventDefault();
    inputUser();
  }
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={new_name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Enter Username"
      />
      <button type="submit">Sumbit</button>
    </form>

  )
}
function App() {
  const [state, setState] = useState(0)
  const [userID, setUserID] = useState('')
  const [questions, setQuestions] = useState([])
  const [showQuestions, setShowQuestions] = useState(false)

  const handleSuccess = (id) => {
    setUserID(id)
    setState(1)
  }

  const handleAddQuestion = (newQuestion) => {
    setQuestions(prev => [...prev, newQuestion])
  }

  return (
    <div>
      <h1 style={{ backgroundColor: "lightblue" }}>Quiz Makers</h1>

      {state === 0 && <EnterUser onSuccess={handleSuccess} />}

      {state === 1 && (
        <div>
          <CreateQuestion onAddQuestion={handleAddQuestion} />

          {questions.length > 0 && (
            <div style={{ marginTop: '24px' }}>
              <button
                onClick={() => setShowQuestions(prev => !prev)}
                style={{
                  padding: '10px 20px',
                  fontSize: '14px',
                  fontWeight: '600',
                  background: '#f5f3ff',
                  color: '#4f46e5',
                  border: '1.5px solid #4f46e5',
                  borderRadius: '8px',
                  cursor: 'pointer',
                }}
              >
                {showQuestions ? '▲ Hide Questions' : '▼ Show Questions'} ({questions.length})
              </button>

              {showQuestions && (
                <div style={{
                  marginTop: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px',
                }}>
                  {questions.map((q, i) => (
                    <div key={i} style={{
                      padding: '16px 20px',
                      background: '#fff',
                      border: '1.5px solid #e0e0e0',
                      borderRadius: '10px',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                    }}>
                      <p style={{ margin: '0 0 10px', fontWeight: '600', color: '#1e1b4b' }}>
                        {i + 1}. {q.question}
                      </p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {q.answers.map((a, j) => (
                          <div key={j} style={{
                            padding: '6px 12px',
                            borderRadius: '6px',
                            fontSize: '14px',
                            background: a.isCorrect ? '#f5f3ff' : 'transparent',
                            color: a.isCorrect ? '#4f46e5' : '#374151',
                            fontWeight: a.isCorrect ? '600' : '400',
                            border: a.isCorrect ? '1px solid #c7d2fe' : '1px solid transparent',
                          }}>
                            {a.isCorrect ? '✓ ' : ''}{a.text}
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
export default App;